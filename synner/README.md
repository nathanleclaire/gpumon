Synner

Synner is a command-line tool that generates synthetic ShareGPT-format data from
a corpus. It reads a Parquet file containing romance literature, splits the text
into narrative chunks, and uses a local Ollama model to transform excerpts into
engaging, turn-based conversations. Synner also includes Git integration commands
to manage dataset changes.

Features
- Synthetic Data Generation: Converts romance literature into ShareGPT
  conversation format.
- Parquet Support: Reads and processes input from Parquet files.
- Ollama Integration: Uses a local Ollama server to generate narrative dialogues.
- Git Integration: Easily create branches and commit changes for dataset updates.


Prerequisites
- Go
- Ollama Server: Running locally (default address: http://localhost:11434).
- Romance Corpus: A Parquet file (default: romance.parquet).

Usage

Build the binary:

```
go build -o synner
```

Generate Synthetic Data

Generate synthetic ShareGPT data from your romance corpus:

```
./synner generate \
  --input-file romance.parquet \
  --out-file datasets/romance/sharegpt_romance.json \
  --model llama2 \
  --ollama-addr http://localhost:11434 \
  --max-examples 1000
```

Git Operations

Create a new Git branch for dataset changes:

./synner branch my-feature-branch

Commit dataset changes with a message:

./synner commit "Generated new synthetic dataset"

Command Flags
 - --input-file: Path to the Parquet file (default: romance.parquet).
 - --out-file: Output JSON file path (default: datasets/romance/sharegpt_romance.json).
 - --model: Local model name in Ollama (default: llama2).
 - --ollama-addr: Ollama server address (default: http://localhost:11434).
 - --max-examples: Maximum number of examples to generate (default: 1000).

Error Handling

Synner logs errors for row reading, text generation, and Git operations. Invalid rows are skipped, and processing continues to ensure maximum dataset coverage.

License

[Specify license details here.]

For questions or contributions, please open an issue or submit a pull request.