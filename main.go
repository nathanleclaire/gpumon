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

type Collector interface {
	Collect(context.Context) ([]GPUData, error)
}

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

// Dynolog logs are all on stderr. We'll parse lines containing "data = { ... }".
type DynologCollector struct {
	cmd     *exec.Cmd
	scanner *bufio.Scanner
}

// Regex capturing JSON after `data =`
var dataRegex = regexp.MustCompile(`data\s*=\s*(\{.*)$`)

func (c *DynologCollector) Start(ctx context.Context) error {
	c.cmd = exec.CommandContext(ctx, "dynolog",
		"--enable_gpu_monitor",
		"--dcgm_lib_path=/lib/x86_64-linux-gnu/libdcgm.so.4",
		"--use_JSON",
	)
	stderr, err := c.cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := c.cmd.Start(); err != nil {
		return err
	}
	// We'll scan stderr instead of stdout
	c.scanner = bufio.NewScanner(stderr)
	return nil
}

// Collect reads one line from stderr that has "data = { ... }" and parses it.
func (c *DynologCollector) Collect(ctx context.Context) ([]GPUData, error) {
	for c.scanner.Scan() {
		line := c.scanner.Text()
		fmt.Println(line) // tee everything to the console
		matches := dataRegex.FindStringSubmatch(line)
		if len(matches) < 2 {
			// No JSON in this line, keep scanning
			continue
		}
		rawJSON := matches[1]
		var j map[string]interface{}
		if err := json.Unmarshal([]byte(rawJSON), &j); err != nil {
			return nil, fmt.Errorf("json parse error: %w", err)
		}
		id := fmt.Sprintf("%v", j["device"])
		memUtil := parseOptionalFloat(j["gpu_memory_utilization"])
		memUsed := int64(memUtil * 1024 * 1024)
		util := parseOptionalFloat(j["sm_active_ratio"])
		return []GPUData{{
			ID:              id,
			Name:            "DCGM-GPU",
			MemoryUsedBytes: memUsed,
			GPUUtilPercent:  int64(util * 100),
		}}, nil
	}
	// If we exit the for-loop, scanner is done or no JSON found.
	if err := c.scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading dynolog stderr: %w", err)
	}
	return nil, fmt.Errorf("no dynolog JSON lines found yet")
}

func parseOptionalFloat(val interface{}) float64 {
	f, _ := strconv.ParseFloat(fmt.Sprintf("%v", val), 64)
	return f
}

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

func registerCallback(m meterWithGauges, c Collector) error {
	_, err := m.meter.RegisterCallback(
		func(ctx context.Context, obs metric.Observer) error {
			slog.Debug("Collecting metrics")
			gpuData, err := c.Collect(ctx)
			if err != nil {
				// Return error so it shows up in logs
				return err
			}
			for _, g := range gpuData {
				attrs := []attribute.KeyValue{
					attribute.String("gpu_id", g.ID),
					attribute.String("gpu_name", g.Name),
				}
				obs.ObserveInt64(m.memGauge, g.MemoryUsedBytes, metric.WithAttributes(attrs...))
				obs.ObserveInt64(m.utilGauge, g.GPUUtilPercent, metric.WithAttributes(attrs...))
			}
			return nil
		},
		m.memGauge,
		m.utilGauge,
	)
	return err
}

func initProvider(ctx context.Context, cfg Config) (func(), error) {
	res, err := resource.New(
		ctx,
		resource.WithAttributes(
			semconv.ServiceName(cfg.ServiceName),
		),
	)
	if err != nil {
		return nil, err
	}
	exp, err := otlpmetricgrpc.New(
		ctx,
		otlpmetricgrpc.WithEndpoint("api.honeycomb.io:443"),
		otlpmetricgrpc.WithHeaders(
			map[string]string{"x-honeycomb-team": cfg.HoneycombKey},
		),
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

func runCollector(ctx context.Context, collector Collector, cfg Config) error {
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
	if err := registerCallback(mwg, collector); err != nil {
		return fmt.Errorf("callback registration error: %w", err)
	}
	slog.Info("Metrics collection running; Ctrl+C to exit.")
	<-ctx.Done()
	return nil
}

// ----------------------------------------------------------------------------
// Cobra commands
// ----------------------------------------------------------------------------

var rootCmd = &cobra.Command{
	Use: "gpu-metrics",
}

var nvidiaSmiCmd = &cobra.Command{
	Use:   "nvidia-smi-poll",
	Short: "Collect GPU metrics via nvidia-smi",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()
		cfg := loadConfig()
		return runCollector(ctx, &NvidiaSMICollector{}, cfg)
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
		return runCollector(ctx, dc, cfg)
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
