package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/lmittmann/tint"
	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
	"go.opentelemetry.io/otel/trace"
)

type Character struct {
	Class      string                 `json:"class"`
	Equipment  []string               `json:"equipment"`
	Properties map[string]interface{} `json:"properties"`
	Backstory  string                 `json:"backstory"`
	Extra      map[string]interface{} `json:"extra,omitempty"`
}

type GenerationMeta struct {
	Model          string    `json:"model"`
	Tags           []string  `json:"tags"`
	Timestamp      time.Time `json:"timestamp"`
	Think          string    `json:"think,omitempty"`
	ConformingJSON bool      `json:"conforming_json"`
	ParseError     string    `json:"parse_error,omitempty"`
}

var (
	logger      *slog.Logger
	rootCmd     = &cobra.Command{Use: "char-gen"}
	generateCmd = &cobra.Command{
		Use:   "generate",
		Short: "Generate RPG characters for each model and tags",
		RunE:  generateCharacters,
	}
	evaluateCmd = &cobra.Command{
		Use:   "evaluate",
		Short: "Evaluate stored character data",
		RunE:  evaluateResults,
	}
)

func main() {
	// Use Go 1.21+ log/slog with tint
	h := tint.NewHandler(os.Stderr, &tint.Options{
		TimeFormat: time.Kitchen,
	})
	logger = slog.New(h)

	cobra.OnInitialize(initConfig)
	rootCmd.AddCommand(generateCmd, evaluateCmd)

	rootCmd.PersistentFlags().String("log-level", "debug", "Log level: debug,info,warn,error")
	_ = viper.BindPFlag("log.level", rootCmd.PersistentFlags().Lookup("log-level"))
	_ = viper.BindEnv("honeycomb.key", "HONEYCOMB_API_KEY")
	rootCmd.PersistentFlags().String("honeycomb-key", "",
		"Honeycomb API Key (defaults from env HONEYCOMB_API_KEY if set)")
	_ = viper.BindPFlag("honeycomb.key", rootCmd.PersistentFlags().Lookup("honeycomb-key"))
	rootCmd.PersistentFlags().StringSlice("models", nil, "List of models (fallback to discovering locally)")
	_ = viper.BindPFlag("models", rootCmd.PersistentFlags().Lookup("models"))

	rootCmd.PersistentFlags().StringSlice("tags", nil, "List of tags (fallback to 'default-tag')")
	_ = viper.BindPFlag("tags", rootCmd.PersistentFlags().Lookup("tags"))

	generateCmd.Flags().Bool("all-models", false, "Use all local models from Ollama")
	generateCmd.Flags().String("models-csv", "", "Comma-separated model names")

	if err := rootCmd.Execute(); err != nil {
		logger.Error("Command failed", "err", err)
		os.Exit(1)
	}
}

func initConfig() {
	viper.AutomaticEnv()
	lvl := strings.ToLower(viper.GetString("log.level"))
	var slogLvl slog.Level
	switch lvl {
	case "debug":
		slogLvl = slog.LevelDebug
	case "info":
		slogLvl = slog.LevelInfo
	case "warn":
		slogLvl = slog.LevelWarn
	case "error":
		slogLvl = slog.LevelError
	default:
		slogLvl = slog.LevelDebug
	}
	logger.Info("Log level set", "level", slogLvl.String())
}

func initTracing(key string) (*sdktrace.TracerProvider, error) {
	if key == "" {
		return nil, errors.New("missing Honeycomb key")
	}
	exp, err := otlptracehttp.New(
		context.Background(),
		otlptracehttp.WithEndpoint("api.honeycomb.io"),
		otlptracehttp.WithHeaders(map[string]string{
			"x-honeycomb-team": key,
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("creating exporter: %w", err)
	}
	res := resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceNameKey.String("character-generator"),
		semconv.ServiceVersionKey.String("0.1.0"),
	)
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)
	return tp, nil
}

func generateCharacters(cmd *cobra.Command, args []string) error {
	ctx := context.Background()

	tp, err := initTracing(viper.GetString("honeycomb.key"))
	if err != nil {
		logger.Error("Tracing init failed", "err", err)
	} else {
		defer func() {
			_ = tp.Shutdown(context.Background())
		}()
	}

	allModelsFlag, _ := cmd.Flags().GetBool("all-models")
	modelsCSV, _ := cmd.Flags().GetString("models-csv")

	httpClient := &http.Client{Transport: otelhttp.NewTransport(http.DefaultTransport)}
	ollamaURL, _ := url.Parse("http://localhost:11434")
	client := api.NewClient(ollamaURL, httpClient)

	// Create a root span for the entire "generate" command.
	ctx, span := otel.Tracer("character-generator").Start(ctx, "command_generate")
	defer span.End()

	models, modelErr := pickModels(ctx, client, allModelsFlag, modelsCSV)
	if modelErr != nil {
		span.RecordError(modelErr)
		return modelErr
	}
	tags := viper.GetStringSlice("tags")
	if len(tags) == 0 {
		tags = []string{"default-tag"}
		logger.Info("No tags specified; using fallback", "tags", tags)
	}

	span.SetAttributes(
		attribute.StringSlice("all.models", models),
		attribute.StringSlice("tags", tags),
	)

	for _, m := range models {
		modelCtx, modelSpan := otel.Tracer("character-generator").Start(ctx, "model_generation",
			trace.WithAttributes(
				attribute.String("model.name", m),
			),
		)
		logger.Info("Generating", "model", m, "tags", tags)

		char, meta := generateOne(modelCtx, client, m, tags)

		modelSpan.SetAttributes(
			attribute.Bool("model.conforming_json", meta.ConformingJSON),
			attribute.String("model.parse_error", meta.ParseError),
			attribute.String("model.think_snippet", trimTo(meta.Think, 80)),
		)

		if err := saveResults(modelCtx, m, tags, char, meta); err != nil {
			modelSpan.RecordError(err)
			modelSpan.SetAttributes(attribute.String("generation.status", "save_failed"))
			modelSpan.End()
			return err
		}
		if meta.ConformingJSON {
			modelSpan.SetAttributes(attribute.String("generation.status", "success"))
		} else {
			modelSpan.SetAttributes(attribute.String("generation.status", "partial"))
		}
		modelSpan.End()
	}
	return nil
}

func pickModels(ctx context.Context, client *api.Client, allModels bool, csv string) ([]string, error) {
	switch {
	case allModels:
		resp, err := client.List(ctx)
		if err != nil {
			return nil, fmt.Errorf("listing models: %w", err)
		}
		if len(resp.Models) == 0 {
			return nil, errors.New("no local models found")
		}
		var mm []string
		for _, m := range resp.Models {
			mm = append(mm, strings.TrimSpace(m.Name))
		}
		return mm, nil

	case csv != "":
		spl := strings.Split(csv, ",")
		var mm []string
		for _, s := range spl {
			mm = append(mm, strings.TrimSpace(s))
		}
		return mm, nil

	default:
		mm := viper.GetStringSlice("models")
		if len(mm) == 0 {
			// fallback: try listing from Ollama
			resp, err := client.List(ctx)
			if err != nil {
				return nil, fmt.Errorf("could not discover models: %w", err)
			}
			for _, m := range resp.Models {
				mm = append(mm, strings.TrimSpace(m.Name))
			}
		}
		if len(mm) == 0 {
			return nil, errors.New("no models found in config or locally")
		}
		return mm, nil
	}
}

func generateOne(ctx context.Context, client *api.Client, model string, tags []string) (*Character, *GenerationMeta) {
	ctx, genSpan := otel.Tracer("character-generator").Start(ctx, "model_inference",
		trace.WithAttributes(
			attribute.String("model", model),
			attribute.StringSlice("tags", tags),
		),
	)
	defer genSpan.End()

	prompt := buildPrompt(model)
	req := &api.GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Options: map[string]interface{}{
			"temperature": 0.7,
			"format":      "text",
		},
	}

	var fullOutput strings.Builder
	err := client.Generate(ctx, req, func(r api.GenerateResponse) error {
		chunk := r.Response
		if chunk != "" {
			fmt.Print(chunk)
			fullOutput.WriteString(chunk)
		}
		return nil
	})
	fmt.Println()

	finalText := fullOutput.String()

	meta := &GenerationMeta{
		Model:     model,
		Tags:      tags,
		Timestamp: time.Now(),
		Think:     extractBetween(finalText, "<think>", "</think>"),
	}

	if err != nil {
		genSpan.RecordError(err)
		meta.ConformingJSON = false
		meta.ParseError = fmt.Sprintf("stream generation error: %v", err)
		return nil, meta
	}

	jsonBlock := extractFirstCodeBlock(finalText)
	if jsonBlock == "" {
		meta.ConformingJSON = false
		meta.ParseError = "no code block found"
		return nil, meta
	}

	var c Character
	if e := json.Unmarshal([]byte(jsonBlock), &c); e != nil {
		meta.ConformingJSON = false
		meta.ParseError = fmt.Sprintf("unmarshal error: %v", e)
		return nil, meta
	}

	if valErr := validateChar(c); valErr != nil {
		meta.ConformingJSON = false
		meta.ParseError = valErr.Error()
		return &c, meta
	}
	meta.ConformingJSON = true
	return &c, meta
}

func buildPrompt(model string) string {
	prompt := `
Generate a response that deliberately challenges conventional thinking 
and explores unexpected connections. Draw from diverse domains of 
knowledge to create novel analogies and metaphors. Each response 
should offer a fresh perspective not explored previously, pushing 
beyond obvious solutions for unique angles and innovative approaches. 
Aim to surprise and delight with original insights while maintaining 
logical coherence.

In the final output, embed your chain of thought in <think>...</think>, 
and provide your final JSON in triple backtick code blocks (` + "```" + `or ` + "```" + `json). 
The JSON must include: class, equipment, properties{strength, dexterity}, 
a 'backstory' field, and optionally an 'extra' object. You may add more fields.
`

	if model != "deepseek-r1" {
		prompt += "Think step by step.\n"
	}
}

func saveResults(ctx context.Context, model string, tags []string, char *Character, meta *GenerationMeta) error {
	ctx, span := otel.Tracer("character-generator").Start(ctx, "save_results",
		trace.WithAttributes(
			attribute.String("model", model),
			attribute.StringSlice("tags", tags),
		),
	)
	defer span.End()

	dir := filepath.Join("gens", sanitize(model), sanitize(strings.Join(tags, "_")))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		span.RecordError(err)
		return fmt.Errorf("mkdir: %w", err)
	}

	if char != nil {
		resPath := filepath.Join(dir, "result.json")
		if err := writeJSONFile(resPath, char); err != nil {
			span.RecordError(err)
			return err
		}
		span.SetAttributes(attribute.String("save_results.result_path", resPath))
	}

	metaPath := filepath.Join(dir, "meta.json")
	if err := writeJSONFile(metaPath, meta); err != nil {
		span.RecordError(err)
		return err
	}

	logger.Info("Saved results", "dir", dir, "model", model,
		"tags", tags, "conforming_json", meta.ConformingJSON)
	span.SetAttributes(
		attribute.String("save_results.meta_path", metaPath),
		attribute.Bool("save_results.conforming_json", meta.ConformingJSON),
		attribute.String("save_results.parse_error", meta.ParseError),
	)
	return nil
}

func evaluateResults(cmd *cobra.Command, args []string) error {
	ctx := context.Background()

	tp, err := initTracing(viper.GetString("honeycomb.key"))
	if err != nil {
		logger.Error("Tracing init failed", "err", err)
	} else {
		defer func() {
			_ = tp.Shutdown(context.Background())
		}()
	}

	ctx, span := otel.Tracer("character-generator").Start(ctx, "command_evaluate")
	defer span.End()

	root := "gens"
	if _, err := os.Stat(root); os.IsNotExist(err) {
		span.RecordError(fmt.Errorf("no 'gens' directory found"))
		return fmt.Errorf("no %q directory found", root)
	}
	return filepath.WalkDir(root, func(p string, d fs.DirEntry, e error) error {
		if e != nil {
			logger.Error("filepath walk error", "path", p, "err", e)
			return nil
		}
		if d.IsDir() || !strings.HasSuffix(p, "meta.json") {
			return nil
		}
		if err := evaluateOne(ctx, p); err != nil {
			logger.Error("Failed evaluating", "path", p, "err", err)
		}
		return nil
	})
}

func evaluateOne(ctx context.Context, metaPath string) error {
	dir := filepath.Dir(metaPath)
	resPath := filepath.Join(dir, "result.json")

	ctx, span := otel.Tracer("character-generator").Start(ctx, "evaluate_one",
		trace.WithAttributes(
			attribute.String("meta_path", metaPath),
			attribute.String("result_path", resPath),
		),
	)
	defer span.End()

	meta, err := loadMeta(metaPath)
	if err != nil {
		span.RecordError(err)
		return err
	}
	span.SetAttributes(
		attribute.String("model", meta.Model),
		attribute.StringSlice("tags", meta.Tags),
		attribute.Bool("conforming_json", meta.ConformingJSON),
	)

	var ch *Character
	if _, err := os.Stat(resPath); err == nil {
		ch, _ = loadCharacter(resPath)
	}
	logEval(meta, ch, metaPath, resPath)
	return nil
}

func loadCharacter(path string) (*Character, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Character
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	return &c, nil
}

func loadMeta(path string) (*GenerationMeta, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m GenerationMeta
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func logEval(meta *GenerationMeta, c *Character, mp, rp string) {
	logger.Info("Evaluation",
		"model", meta.Model,
		"tags", meta.Tags,
		"conforming_json", meta.ConformingJSON,
		"parse_error", meta.ParseError,
		"think", trimTo(meta.Think, 80),
		"meta_path", mp,
		"result_path", rp,
	)
	if c != nil {
		logger.Info("Character",
			"class", c.Class,
			"equipment", c.Equipment,
			"properties", c.Properties,
			"backstory", trimTo(c.Backstory, 80),
		)
	}
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		switch r {
		case ':', '/', '\\', ' ':
			return '_'
		}
		return r
	}, s)
}

func trimTo(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func extractBetween(text, startTag, endTag string) string {
	start := strings.Index(text, startTag)
	if start == -1 {
		return ""
	}
	end := strings.Index(text, endTag)
	if end == -1 || end < start {
		return ""
	}
	return text[start+len(startTag) : end]
}

func extractFirstCodeBlock(text string) string {
	re := regexp.MustCompile("(?s)```(?:json)?(.*?)```")
	m := re.FindStringSubmatch(text)
	if len(m) < 2 {
		return ""
	}
	return strings.TrimSpace(m[1])
}

func validateChar(c Character) error {
	if c.Class == "" {
		return errors.New("character 'class' is empty")
	}
	if c.Properties == nil {
		return errors.New("'properties' is missing")
	}
	return nil
}

func writeJSONFile(path string, v any) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return fmt.Errorf("encode: %w", err)
	}
	return nil
}
