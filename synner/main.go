package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/lmittmann/tint"
	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/reader"
	"github.com/xitongsys/parquet-go/source"
)

type ShareGPTTurn struct {
	From  string `json:"from"`
	Value string `json:"value"`
}

type ShareGPTData struct {
	Conversations [][]ShareGPTTurn `json:"conversations"`
}

type DataSource interface {
	NextRow() (string, error)
	Close() error
}

type parquetSource struct {
	pr  *reader.ParquetReader
	f   source.ParquetFile
	cur int64
	max int64
}

type RomanceRow struct {
	URL  string `parquet:"name=url,type=BYTE_ARRAY,convertedtype=UTF8,repetitiontype=OPTIONAL"`
	Text string `parquet:"name=text,type=BYTE_ARRAY,convertedtype=UTF8,repetitiontype=OPTIONAL"`
}

func (p *parquetSource) NextRow() (string, error) {
	if p.cur >= p.max {
		return "", io.EOF
	}
	rows, err := p.pr.ReadByNumber(1)
	if err != nil {
		return "", fmt.Errorf("failed to read row: %w", err)
	}
	p.cur++
	if len(rows) == 0 {
		return "", io.EOF
	}
	rr, ok := rows[0].(RomanceRow)
	if !ok {
		return "", fmt.Errorf("invalid row type: %T", rows[0])
	}
	if rr.Text == "" {
		return "", fmt.Errorf("empty text field in row")
	}
	return rr.Text, nil
}

func (p *parquetSource) Close() error {
	p.pr.ReadStop()
	return p.f.Close()
}

func main() {
	// Force debug-level logging so partial chunks appear.
	logger := slog.New(tint.NewHandler(os.Stderr, &tint.Options{
		TimeFormat: "15:04",
		Level:      slog.LevelDebug, // Ensure debug logs are displayed
	}))
	rootCmd := &cobra.Command{Use: "synner"}
	rootCmd.AddCommand(
		newGenerateCmd(logger),
		newBranchCmd(logger),
		newCommitCmd(logger),
	)
	if err := rootCmd.Execute(); err != nil {
		logger.Error("command failed", "err", err)
		os.Exit(1)
	}
}

func newGenerateCmd(logger *slog.Logger) *cobra.Command {
	var inFile, outFile, modelName, ollamaAddr string
	var maxExamples int
	cmd := &cobra.Command{
		Use:   "generate",
		Short: "Generate synthetic ShareGPT-format data from a romance corpus",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenerate(logger, inFile, outFile, modelName,
				ollamaAddr, maxExamples)
		},
	}
	cmd.Flags().StringVar(&inFile, "input-file",
		"romance.parquet", "Parquet file")
	cmd.Flags().StringVar(&outFile, "out-file",
		filepath.Join("datasets", "romance", "sharegpt_romance.json"),
		"Output JSON")
	cmd.Flags().StringVar(&modelName, "model",
		"llama2", "Local model name in Ollama")
	cmd.Flags().StringVar(&ollamaAddr, "ollama-addr",
		"http://localhost:11434", "Ollama server address")
	cmd.Flags().IntVar(&maxExamples, "max-examples",
		1000, "Max examples to generate")
	return cmd
}

func newBranchCmd(logger *slog.Logger) *cobra.Command {
	return &cobra.Command{
		Use:   "branch [branch-name]",
		Short: "Create a git branch for dataset changes",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGitCommand(logger, "checkout", "-b", args[0])
		},
	}
}

func newCommitCmd(logger *slog.Logger) *cobra.Command {
	return &cobra.Command{
		Use:   "commit [msg]",
		Short: "Commit dataset changes",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := runGitCommand(logger, "add", "datasets"); err != nil {
				return err
			}
			return runGitCommand(logger, "commit", "-m", args[0])
		},
	}
}

func runGenerate(logger *slog.Logger, inFile, outFile, model, ollamaAddr string, maxEx int) error {
	ds, err := openParquetSource(inFile)
	if err != nil {
		return err
	}
	defer ds.Close()

	allRows := readAllRows(ds, logger)
	if len(allRows) == 0 {
		return errors.New("no valid rows found")
	}
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(allRows), func(i, j int) {
		allRows[i], allRows[j] = allRows[j], allRows[i]
	})

	ch := newParagraphChunker(3, 200)
	client := &http.Client{}
	c := api.NewClient(mustParseURL(ollamaAddr), client)
	existing, _ := loadShareGPT(outFile)

	var totalChunks int
	for _, row := range allRows {
		totalChunks += len(ch.Split(row))
	}
	logger.Info("Starting generation",
		"totalBooks", len(allRows),
		"totalChunks", totalChunks)

	ctx := context.Background()
	var count, chunkSoFar int
	for i, row := range allRows {
		if count >= maxEx {
			break
		}
		logger.Info("Processing book",
			"index", i+1,
			"totalBooks", len(allRows),
			"preview", trimTo(row, 80))

		chunks := ch.Split(row)
		for j, chunk := range chunks {
			chunkSoFar++
			if count >= maxEx {
				break
			}
			logger.Info("Generating chunk",
				"chunkIndex", j+1,
				"chunksInBook", len(chunks),
				"globalChunkIndex", chunkSoFar,
				"totalChunks", totalChunks)

			resp, err := generateChatOllama(ctx, c, model, chunk, logger)
			if err != nil {
				logger.Error("ollama generate error",
					"chunk_preview", trimTo(chunk, 60),
					"err", err)
				continue
			}
			if len(resp) > 0 {
				existing.Conversations = append(existing.Conversations, resp)
				count++
			}
		}
	}

	if err := saveShareGPT(outFile, existing); err != nil {
		return err
	}
	logger.Info("Generation complete",
		"output", outFile,
		"count", count,
		"totalRows", len(allRows))
	return nil
}

func readAllRows(ds DataSource, logger *slog.Logger) []string {
	var rows []string
	for {
		row, err := ds.NextRow()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			logger.Error("Row read error", "err", err)
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

func openParquetSource(path string) (DataSource, error) {
	f, err := local.NewLocalFileReader(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open parquet file: %w", err)
	}
	pr, err := reader.NewParquetReader(f, new(RomanceRow), 4)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to create parquet reader: %w", err)
	}
	max := pr.GetNumRows()
	if max == 0 {
		f.Close()
		pr.ReadStop()
		return nil, fmt.Errorf("parquet file contains no rows")
	}
	return &parquetSource{pr: pr, f: f, max: max}, nil
}

type paragraphChunker struct {
	paragraphsPerChunk int
	minChunkLength     int
}

func newParagraphChunker(paragraphsPerChunk, minChunkLength int) *paragraphChunker {
	if paragraphsPerChunk <= 0 {
		paragraphsPerChunk = 3
	}
	if minChunkLength <= 0 {
		minChunkLength = 100
	}
	return &paragraphChunker{
		paragraphsPerChunk: paragraphsPerChunk,
		minChunkLength:     minChunkLength,
	}
}

func (p *paragraphChunker) Split(row string) []string {
	paragraphs := strings.Split(row, "\n")
	var clean []string
	for _, pp := range paragraphs {
		t := strings.TrimSpace(pp)
		if t != "" {
			clean = append(clean, t)
		}
	}
	if len(clean) == 0 {
		return nil
	}
	var chunks []string
	var current []string
	for i, para := range clean {
		current = append(current, para)
		if len(current) >= p.paragraphsPerChunk || i == len(clean)-1 {
			chunk := strings.Join(current, "\n\n")
			if len(chunk) >= p.minChunkLength {
				chunks = append(chunks, chunk)
			}
			current = nil
			if i < len(clean)-1 {
				if len(current) == 0 && i > 0 {
					current = append(current, clean[i])
				}
			}
		}
	}
	return chunks
}

// generateChatOllama logs each partial chunk from Ollama as it's received.
func generateChatOllama(ctx context.Context, c *api.Client,
	model, snippet string, _ *slog.Logger) ([]ShareGPTTurn, error) {

	prompt := fmt.Sprintf(`
You are an expert narrative synthesizer tasked with transforming a romance
literature excerpt into an immersive and suspenseful experience. Your goal is
to create a turn-based conversation between a narrator gpt (who will outline the
scene and perform the dialogue of NPCs) and the human (who will be the human user
in the final trained chatbot).

Your task is to generate an emotionally authentic narrator/user roleplay based
on the given literature excerpt:

<literature>
%q
</literature>

Key Requirements:
- Emphasize a **romantic narrative**.
- Attempt to understand the characters' names, relationships, and the context of the story.
- Maintain consistent character voices and narrative flow throughout the conversation.
- Include subtle relationship dynamics and tension.
- Incorporate occasional actions or non-verbal cues in parentheses.
- Generate five conversation turns, with the gpt response's length ALWAYS being
  about **three to five paragraphs** of AT LEAST three sentences each, and the
  user's input at about one or two sentences.
- Vary the length of responses organically.
- Human will always go first per-turn, then GPT.
- Human will always be the main character from the chunk of literature. Make a best
  guess as you walk through the excerpt who the main character is to insert them
  as.

Output the conversation in the following JSON structure, enclosed in <json> tags.
**YOUR RESPONSE MUST INCLUDE THESE TAGS**.

<json>
{
	"conversations": [
	[
		{"from": "human", "value": "dialogue"},
		{"from": "gpt",   "value": "response"}
	]
	]
}
</json>

Example:

<literature>
Elizabeth could not help but observe Mr. Darcy across the crowded ballroom. His
tall figure cut an imposing silhouette against the candlelit walls, and though
he maintained his usual stern countenance, she caught his eyes following her
movements more than once. Their last heated argument about her sister's
engagement to Mr. Bingley still burned fresh in her mind.
</literature>

Expected Output:

<json>
{
"conversations": [
[
{"from": "human", "value": "I want to approach Mr. Darcy, but after our last argument, I'm hesitant. Perhaps I should simply observe him from afar for now."},
{"from": "gpt", "value": "The grandiose ballroom sparkles with candlelight, casting dramatic shadows across the elaborately decorated walls. Mr. Darcy stands apart from the crowd, his commanding presence drawing attention even in his solitude. Though he maintains his characteristic stoic expression, his dark eyes seem to find you with remarkable frequency among the swirling dancers and chattering guests.\n\nMrs. Bennet's shrill voice carries across the room as she loudly proclaims the virtues of your sister Jane to anyone who will listen. The celebration of her engagement to Mr. Bingley has set all of Meryton abuzz with excitement and speculation.\n\nYou notice Mr. Darcy's jaw tighten almost imperceptibly when your eyes meet briefly across the room. The memory of his harsh words about your family's social standing and his interference in Jane's relationship with Mr. Bingley still stings, though something in his gaze now seems different – perhaps tinged with regret?"}
],
[
{"from": "human", "value": "I shall not let him intimidate me. I straighten my posture and meet his gaze directly."},
{"from": "gpt", "value": "A subtle spark of approval seems to flicker in Mr. Darcy's eyes at your display of fortitude. He inclines his head ever so slightly in acknowledgment, the gesture barely perceptible to any but the most attentive observer. The string quartet begins a new piece, its gentle melody weaving through the excited murmurs of the assembled company.\n\nMr. Bingley approaches his friend, speaking in animated tones that contrast sharply with Mr. Darcy's reserved demeanor. Though you cannot hear their words, you see Mr. Darcy's attention remain fixed in your direction even as he responds to his companion.\n\n'Oh, Lizzy!' your younger sister Kitty appears at your elbow, practically bouncing with excitement. 'Is it not thrilling? And to think, you might have had your own wealthy suitor if you hadn't been so sharp with Mr. Darcy!' (She giggles, oblivious to the complexity of the situation)"}
]
]
}
</json>
`, snippet)
	req := &api.GenerateRequest{
		Model:   model,
		Prompt:  prompt,
		Options: map[string]interface{}{"temperature": 0.7},
	}

	var full strings.Builder
	tokenCh := make(chan string, 32)
	done := make(chan struct{})

	const (
		minDelay = 10 * time.Millisecond
		maxDelay = 50 * time.Millisecond
	)

	// Printing goroutine with dynamic speed
	go func() {
		defer close(done)
		for t := range tokenCh {
			// How much of the channel is filled? 0.0 => empty, 1.0 => full
			usage := float64(len(tokenCh)) / float64(cap(tokenCh))

			// Scale delay so it's smaller (faster) if usage is high
			delay := time.Duration(
				float64(minDelay) +
					(1.0-usage)*float64(maxDelay-minDelay),
			)
			for _, r := range t {
				fmt.Printf("%c", r)
				time.Sleep(delay)
			}
		}
	}()

	err := c.Generate(ctx, req, func(r api.GenerateResponse) error {
		if r.Response != "" {
			tokenCh <- r.Response
			full.WriteString(r.Response)
		}
		return nil
	})

	close(tokenCh)
	<-done

	fmt.Print("\n\n")

	if err != nil {
		return nil, err
	}

	body := full.String()
	jsonBlock := extractBetween(body, "<json>", "</json>")
	if jsonBlock == "" {
		return nil, errors.New("no <json> block found")
	}
	var outer struct {
		Conversations [][]ShareGPTTurn `json:"conversations"`
	}
	if e := json.Unmarshal([]byte(jsonBlock), &outer); e != nil {
		return nil, e
	}
	if len(outer.Conversations) == 0 {
		return nil, errors.New("no conversation data found")
	}
	return outer.Conversations[0], nil
}

func extractBetween(s, start, end string) string {
	i := strings.Index(s, start)
	if i == -1 {
		return ""
	}
	j := strings.Index(s[i+len(start):], end)
	if j == -1 {
		return ""
	}
	return s[i+len(start) : i+len(start)+j]
}

func loadShareGPT(path string) (*ShareGPTData, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return &ShareGPTData{}, nil
	}
	var d ShareGPTData
	if e := json.Unmarshal(b, &d); e != nil {
		return nil, e
	}
	return &d, nil
}

func saveShareGPT(path string, d *ShareGPTData) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(d)
}

func runGitCommand(logger *slog.Logger, subcmd string, args ...string) error {
	cmd := exec.Command("git", append([]string{subcmd}, args...)...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		logger.Error("git error", "stderr", stderr.String())
		return err
	}
	logger.Info("git "+subcmd+" ok", "args", args)
	return nil
}

func mustParseURL(s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil {
		panic(err)
	}
	return u
}

func trimTo(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
