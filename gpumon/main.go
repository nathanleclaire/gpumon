package main

import (
	"bufio"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/metric"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

type Config struct {
	ServiceName    string
	HoneycombKey   string
	MetricInterval time.Duration
}

type GPUData struct {
	ID              string
	Name            string
	MemoryUsedBytes int64
	GPUUtilPercent  int64
}

// DynologData now matches the JSON types exactly. For numeric fields in quotes,
// we use `,string` so Unmarshal succeeds. For numeric fields without quotes, we
// omit `,string`.
type DynologData struct {
	DCGMError           int64   `json:"dcgm_error"`
	Device              int64   `json:"device"`
	FP16Active          float64 `json:"fp16_active,string"`
	FP32Active          float64 `json:"fp32_active,string"`
	FP64Active          float64 `json:"fp64_active,string"`
	GPUFreqMHz          float64 `json:"gpu_frequency_mhz"`
	GPUMemoryUtil       float64 `json:"gpu_memory_utilization"`
	GPUPowerDraw        float64 `json:"gpu_power_draw,string"`
	GraphicsActiveRatio float64 `json:"graphics_engine_active_ratio,string"`
	HbmMemBWUtil        float64 `json:"hbm_mem_bw_util,string"`
	NvlinkRxBytes       int64   `json:"nvlink_rx_bytes"`
	NvlinkTxBytes       int64   `json:"nvlink_tx_bytes"`
	PcieRxBytes         int64   `json:"pcie_rx_bytes"`
	PcieTxBytes         int64   `json:"pcie_tx_bytes"`
	SmActiveRatio       float64 `json:"sm_active_ratio,string"`
	SmOccupancy         float64 `json:"sm_occupancy,string"`
	TensorcoreActive    float64 `json:"tensorcore_active,string"`
}

// -----------------------------------------------------------------------------
// NVIDIA SMI Collector
// -----------------------------------------------------------------------------

type NvidiaSMICollector struct{}

func (c *NvidiaSMICollector) Collect(ctx context.Context) ([]GPUData, error) {
	out, err := exec.CommandContext(ctx, "nvidia-smi", "-q", "-x").Output()
	if err != nil {
		return nil, fmt.Errorf("exec error: %w", err)
	}
	var smiLog struct {
		GPUs []struct {
			ID          string `xml:"id,attr"`
			ProductName string `xml:"product_name"`
			FBMemory    struct {
				Used string `xml:"used"`
			} `xml:"fb_memory_usage"`
			Utilization struct {
				GPUUtil string `xml:"gpu_util"`
			} `xml:"utilization"`
		} `xml:"gpu"`
	}
	if err := xml.Unmarshal(out, &smiLog); err != nil {
		return nil, fmt.Errorf("unmarshal error: %w", err)
	}
	var results []GPUData
	for _, g := range smiLog.GPUs {
		mem, _ := parseMemory(g.FBMemory.Used)
		util, _ := parsePercentage(g.Utilization.GPUUtil)
		results = append(results, GPUData{
			ID:              g.ID,
			Name:            g.ProductName,
			MemoryUsedBytes: mem,
			GPUUtilPercent:  util,
		})
	}
	return results, nil
}

// -----------------------------------------------------------------------------
// Dynolog Collector
// -----------------------------------------------------------------------------

// Regex capturing JSON after `data =`
var dataRegex = regexp.MustCompile(`data\s*=\s*(\{.*)$`)

type DynologCollector struct {
	cmd     *exec.Cmd
	scanner *bufio.Scanner
}

func (c *DynologCollector) Start(ctx context.Context) error {
	c.cmd = exec.CommandContext(ctx, "dynolog",
		"--enable_gpu_monitor",
		"--dcgm_lib_path=/lib/x86_64-linux-gnu/libdcgm.so.4",
		"--use_JSON",
		"--dcgm_reporting_interval_s",
		"1",
	)
	stderr, err := c.cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := c.cmd.Start(); err != nil {
		return err
	}
	c.scanner = bufio.NewScanner(stderr)
	return nil
}

func (c *DynologCollector) Collect(ctx context.Context) (DynologData, error) {
	for c.scanner.Scan() {
		line := c.scanner.Text()
		fmt.Println(line) // tee entire line to console
		if m := dataRegex.FindStringSubmatch(line); len(m) >= 2 {
			var raw DynologData
			if err := json.Unmarshal([]byte(m[1]), &raw); err != nil {
				return DynologData{}, err
			}
			return raw, nil
		}
	}
	if err := c.scanner.Err(); err != nil {
		return DynologData{}, err
	}
	return DynologData{}, fmt.Errorf("no dynolog JSON lines found yet")
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

func parsePercentage(val string) (int64, error) {
	s := strings.ReplaceAll(val, "%", "")
	s = strings.TrimSpace(s)
	return strconv.ParseInt(s, 10, 64)
}

func parseMemory(val string) (int64, error) {
	s := strings.ReplaceAll(val, "MiB", "")
	s = strings.TrimSpace(s)
	num, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, err
	}
	return num * 1024 * 1024, nil
}

// -----------------------------------------------------------------------------
// Meter / Gauges
// -----------------------------------------------------------------------------

type meterWithGauges struct {
	meter     metric.Meter
	memGauge  metric.Int64ObservableGauge
	utilGauge metric.Int64ObservableGauge
}

func newMeterWithGauges(m metric.Meter) (meterWithGauges, error) {
	memG, err := m.Int64ObservableGauge("gpu.memory_used_bytes")
	if err != nil {
		return meterWithGauges{}, err
	}
	utilG, err := m.Int64ObservableGauge("gpu.utilization_percent")
	if err != nil {
		return meterWithGauges{}, err
	}
	return meterWithGauges{m, memG, utilG}, nil
}

// registerDynologCallback sets up instruments matching DynologData fields.
func registerDynologCallback(m metric.Meter, c *DynologCollector) error {
	dcgmErrGauge, _ := m.Int64ObservableGauge("dcgm.error")
	nvlinkRxGauge, _ := m.Int64ObservableGauge("dcgm.nvlink_rx_bytes")
	nvlinkTxGauge, _ := m.Int64ObservableGauge("dcgm.nvlink_tx_bytes")
	pcieRxGauge, _ := m.Int64ObservableGauge("dcgm.pcie_rx_bytes")
	pcieTxGauge, _ := m.Int64ObservableGauge("dcgm.pcie_tx_bytes")
	fp16Gauge, _ := m.Float64ObservableGauge("dcgm.fp16_active_ratio")
	fp32Gauge, _ := m.Float64ObservableGauge("dcgm.fp32_active_ratio")
	fp64Gauge, _ := m.Float64ObservableGauge("dcgm.fp64_active_ratio")
	freqGauge, _ := m.Float64ObservableGauge("dcgm.gpu_frequency_mhz")
	memUtilGauge, _ := m.Float64ObservableGauge("dcgm.gpu_memory_util")
	powerGauge, _ := m.Float64ObservableGauge("dcgm.gpu_power_draw_watts")
	gfxRatioGauge, _ := m.Float64ObservableGauge("dcgm.graphics_engine_active_ratio")
	hbmGauge, _ := m.Float64ObservableGauge("dcgm.hbm_mem_bw_util")
	smActiveGauge, _ := m.Float64ObservableGauge("dcgm.sm_active_ratio")
	smOccGauge, _ := m.Float64ObservableGauge("dcgm.sm_occupancy_ratio")
	tensorGauge, _ := m.Float64ObservableGauge("dcgm.tensorcore_active_ratio")

	_, err := m.RegisterCallback(
		func(ctx context.Context, obs metric.Observer) error {
			slog.Debug("Collecting dynolog metrics")
			data, err := c.Collect(ctx)
			if err != nil {
				return err
			}
			// Convert device int64 -> string for attribute
			attrs := []attribute.KeyValue{
				attribute.String("gpu_id", fmt.Sprintf("%d", data.Device)),
			}
			obs.ObserveInt64(dcgmErrGauge, data.DCGMError, metric.WithAttributes(attrs...))
			obs.ObserveInt64(nvlinkRxGauge, data.NvlinkRxBytes, metric.WithAttributes(attrs...))
			obs.ObserveInt64(nvlinkTxGauge, data.NvlinkTxBytes, metric.WithAttributes(attrs...))
			obs.ObserveInt64(pcieRxGauge, data.PcieRxBytes, metric.WithAttributes(attrs...))
			obs.ObserveInt64(pcieTxGauge, data.PcieTxBytes, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(fp16Gauge, data.FP16Active, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(fp32Gauge, data.FP32Active, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(fp64Gauge, data.FP64Active, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(freqGauge, data.GPUFreqMHz, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(memUtilGauge, data.GPUMemoryUtil, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(powerGauge, data.GPUPowerDraw, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(gfxRatioGauge, data.GraphicsActiveRatio, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(hbmGauge, data.HbmMemBWUtil, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(smActiveGauge, data.SmActiveRatio, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(smOccGauge, data.SmOccupancy, metric.WithAttributes(attrs...))
			obs.ObserveFloat64(tensorGauge, data.TensorcoreActive, metric.WithAttributes(attrs...))
			return nil
		},
		dcgmErrGauge, nvlinkRxGauge, nvlinkTxGauge, pcieRxGauge, pcieTxGauge,
		fp16Gauge, fp32Gauge, fp64Gauge, freqGauge, memUtilGauge,
		powerGauge, gfxRatioGauge, hbmGauge, smActiveGauge, smOccGauge,
		tensorGauge,
	)
	return err
}

// -----------------------------------------------------------------------------
// OTel Provider Setup
// -----------------------------------------------------------------------------

func initProvider(ctx context.Context, cfg Config) (func(), error) {
	res, err := resource.New(
		ctx,
		resource.WithAttributes(semconv.ServiceName(cfg.ServiceName)),
	)
	if err != nil {
		return nil, err
	}
	exp, err := otlpmetricgrpc.New(
		ctx,
		otlpmetricgrpc.WithEndpoint("api.honeycomb.io:443"),
		otlpmetricgrpc.WithHeaders(map[string]string{"x-honeycomb-team": cfg.HoneycombKey}),
	)
	if err != nil {
		return nil, err
	}
	prov := sdkmetric.NewMeterProvider(
		sdkmetric.WithResource(res),
		sdkmetric.WithReader(
			sdkmetric.NewPeriodicReader(exp, sdkmetric.WithInterval(cfg.MetricInterval)),
		),
	)
	otel.SetMeterProvider(prov)
	return func() {
		if err := prov.Shutdown(ctx); err != nil {
			slog.Error("shutdown error", "error", err)
		}
	}, nil
}

// -----------------------------------------------------------------------------
// Runners
// -----------------------------------------------------------------------------

func runNvidiaSmiCollector(ctx context.Context, cfg Config) error {
	shutdown, err := initProvider(ctx, cfg)
	if err != nil {
		return fmt.Errorf("init error: %w", err)
	}
	defer shutdown()

	m := otel.Meter("gpu-metrics")
	mwg, err := newMeterWithGauges(m)
	if err != nil {
		return fmt.Errorf("gauge creation error: %w", err)
	}
	_, err = m.RegisterCallback(func(ctx context.Context, obs metric.Observer) error {
		slog.Debug("Collecting nvidia-smi metrics")
		data, err := (&NvidiaSMICollector{}).Collect(ctx)
		if err != nil {
			return err
		}
		for _, g := range data {
			attrs := []attribute.KeyValue{
				attribute.String("gpu_id", g.ID),
				attribute.String("gpu_name", g.Name),
			}
			obs.ObserveInt64(mwg.memGauge, g.MemoryUsedBytes, metric.WithAttributes(attrs...))
			obs.ObserveInt64(mwg.utilGauge, g.GPUUtilPercent, metric.WithAttributes(attrs...))
		}
		return nil
	}, mwg.memGauge, mwg.utilGauge)
	if err != nil {
		return fmt.Errorf("callback registration error: %w", err)
	}
	slog.Info("nvidia-smi metrics collection running; Ctrl+C to exit.")
	<-ctx.Done()
	return nil
}

func runDynologCollector(ctx context.Context, cfg Config, dc *DynologCollector) error {
	shutdown, err := initProvider(ctx, cfg)
	if err != nil {
		return fmt.Errorf("init error: %w", err)
	}
	defer shutdown()

	m := otel.Meter("gpu-metrics")
	if err := registerDynologCallback(m, dc); err != nil {
		return fmt.Errorf("callback registration error: %w", err)
	}
	slog.Info("dynolog metrics collection running; Ctrl+C to exit.")
	<-ctx.Done()
	return nil
}

// -----------------------------------------------------------------------------
// Cobra commands
// -----------------------------------------------------------------------------

var rootCmd = &cobra.Command{
	Use: "gpu-metrics",
}

var nvidiaSmiCmd = &cobra.Command{
	Use:   "nvidia-smi-poll",
	Short: "Collect GPU metrics via nvidia-smi",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()
		return runNvidiaSmiCollector(ctx, loadConfig())
	},
}

var dynologCmd = &cobra.Command{
	Use:   "dynolog-poll",
	Short: "Collect GPU metrics via dynolog JSON (on stderr)",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()
		cfg := loadConfig()
		dc := &DynologCollector{}
		if err := dc.Start(ctx); err != nil {
			return fmt.Errorf("start dynolog: %w", err)
		}
		return runDynologCollector(ctx, cfg, dc)
	},
}

func loadConfig() Config {
	return Config{
		ServiceName:    viper.GetString("service_name"),
		HoneycombKey:   viper.GetString("honeycomb_key"),
		MetricInterval: 15 * time.Second,
	}
}

func main() {
	viper.SetDefault("service_name", "gpu-mon")
	viper.BindEnv("honeycomb_key", "HONEYCOMB_API_KEY")

	rootCmd.AddCommand(nvidiaSmiCmd, dynologCmd)
	if err := rootCmd.Execute(); err != nil {
		slog.Error("command error", "error", err)
		os.Exit(1)
	}
}
