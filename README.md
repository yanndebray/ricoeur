# ricoeur

**A local-first archive, search, and intelligence engine for your LLM conversation history.**

*Named after Paul Ricoeur, whose work on narrative identity argued that we understand ourselves through the stories we construct from our lived experience. ricoeur reconstructs the narrative of your intellectual life from thousands of AI conversations.*

```
██████╗ ██╗ ██████╗ ██████╗ ███████╗██╗   ██╗██████╗
██╔══██╗██║██╔════╝██╔═══██╗██╔════╝██║   ██║██╔══██╗
██████╔╝██║██║     ██║   ██║█████╗  ██║   ██║██████╔╝
██╔══██╗██║██║     ██║   ██║██╔══╝  ██║   ██║██╔══██╗
██║  ██║██║╚██████╗╚██████╔╝███████╗╚██████╔╝██║  ██║
╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝
              ──────────────────────────
                your conversation archive
```

> The wordmark above is painted on as a brief opening animation when you launch
> `ricoeur tui`. Press any key to skip it, or set `RICOEUR_NO_SPLASH=1` to disable.

## Quickstart

```bash
# Install with uv
uv sync

# Initialize the database
uv run ricoeur init

# Import your ChatGPT export
uv run ricoeur import chatgpt ~/Downloads/chatgpt-export/conversations.json

# ...or your Claude export (Settings > Privacy > Export data)
uv run ricoeur import claude ~/Downloads/claude-export/conversations.json

# Build the intelligence layer (language detection, embeddings, analytics)
uv run ricoeur index

# See what you've got
uv run ricoeur stats

# Search your history
uv run ricoeur search "thermal simulation"
```

## Commands

| Command | Description |
|---------|-------------|
| `ricoeur init` | Initialize database and config at `~/.ricoeur/` |
| `ricoeur import chatgpt <path>` | Import from ChatGPT export (.json or .zip) |
| `ricoeur import claude <path>` | Import from Claude export (.json or .zip) |
| `ricoeur search <query>` | Search across all conversations (hybrid by default) |
| `ricoeur show <id>` | Display a conversation with formatting |
| `ricoeur stats` | Analytics dashboard |
| `ricoeur tui` | Interactive terminal UI to browse and search |
| `ricoeur index` | Build intelligence layer (languages, embeddings, analytics) |
| `ricoeur config show` | Print current configuration |
| `ricoeur config set <key> <value>` | Update a config value |

## Search

ricoeur supports three search modes:

| Mode | Flag | How it works |
|------|------|-------------|
| **Hybrid** | *(default)* | Combines keyword + semantic via Reciprocal Rank Fusion (RRF) |
| **Keyword** | `--keyword` | FTS5 full-text search with BM25 ranking |
| **Semantic** | `--semantic` | Cosine similarity against pre-computed embeddings |

When embeddings are available (after `ricoeur index`), search automatically uses hybrid mode. If no embeddings exist, it falls back to keyword search. Flags like `--code` or `--role` also force keyword mode since they rely on FTS5.

```bash
# Hybrid search (default — combines keyword + semantic)
ricoeur search "deployment strategies"

# Force keyword-only (FTS5)
ricoeur search "streamlit dashboard" --keyword

# Force semantic-only (cosine similarity)
ricoeur search "how to containerize apps" --semantic

# Filter by platform, language, date
ricoeur search "error fix" --platform chatgpt --lang en
ricoeur search "strategie marketing" --lang fr --since 2025-01-01

# Search only in code blocks (auto-uses keyword mode)
ricoeur search "import pandas" --code

# More output formats coming soon: json, full, ids
```

### Why semantic search?

Keyword search only finds exact word matches. Semantic search finds **conceptually related** conversations — even when the exact words don't appear.

**Query: "how to containerize applications"**

```
$ ricoeur search "how to containerize applications" --keyword
Found 3 results for "how to containerize applications" (keyword)

$ ricoeur search "how to containerize applications" --semantic
Found 20 results for "how to containerize applications" (semantic)
 #   Score    Date        Title
 1   0.5515   2025-07-29  Free docker deployment options
 2   0.5152   2026-01-28  Secure Clawdbot Setup
 3   0.5081   2025-03-11  Docker noVNC Setup
 4   0.5014   2025-09-21  HTML upload and serve
 5   0.4782   2022-12-27  Wasm vs Container Comparison
 ...
```

Keyword found 3 results matching the literal words. Semantic found 20 — including Docker, Wasm, and container deployment conversations that never mention "containerize applications".

## Terminal UI

Prefer browsing interactively? Launch the TUI:

```bash
# Install the optional dependency
uv sync --extra tui

# Launch
uv run ricoeur tui
```

It opens to your most recent conversations. Type a query and press **Enter** to
keyword-search; clear the box and press Enter to return to the recent list.
Select a row (Enter) to read the full conversation, rendered as Markdown.

| Key | Action |
|-----|--------|
| `/` | Focus the search box |
| `Enter` | Search (in box) / open conversation (in list) |
| `↑` `↓` `j` `k` | Move / scroll |
| `Esc` | Back to the list |
| `q` | Quit |

## Index

After importing, build the intelligence layer:

```bash
# Run all layers: language detection, embeddings, analytics
ricoeur index

# Second run skips what's already cached
ricoeur index

# Force a full rebuild
ricoeur index --force

# Run specific layers only
ricoeur index --embeddings
ricoeur index --analytics

# Use a different embedding model
ricoeur index --embed-model ollama:nomic-embed-text
ricoeur index --embed-model st:all-MiniLM-L6-v2 --device cpu
```

Each layer requires its optional extra:

| Layer | Extra | What it does |
|-------|-------|-------------|
| Languages | `langdetect` | Detects language per conversation (stored in DB) |
| Embeddings | `embeddings` | Generates sentence-transformer vectors (`~/.ricoeur/embeddings/`) |
| Analytics | `analytics` | Exports conversations & messages to Parquet (`~/.ricoeur/analytics/`) |

```bash
# Install all index dependencies at once
uv sync --extra langdetect --extra embeddings --extra analytics
```

## Import options

```bash
# Re-import safely (updates existing, adds new, never deletes)
ricoeur import chatgpt conversations.json --update

# Dry run — parse and validate without writing
ricoeur import chatgpt conversations.json --dry-run

# Only import recent conversations
ricoeur import chatgpt conversations.json --since 2025-01-01

# The same flags work for Claude exports
ricoeur import claude conversations.json --update
```

ChatGPT and Claude exports are both supported. Each platform ships a
`conversations.json` (sometimes inside a `.zip`) — point ricoeur at either the
JSON file or the zip and it will find the conversations.

> **Coming soon:** Gemini and custom JSON imports.

## Optional extras

Install additional capabilities as needed:

```bash
# Language detection
uv sync --extra langdetect

# Semantic search with sentence-transformers
uv sync --extra embeddings

# Topic modeling with BERTopic (coming soon)
uv sync --extra topics

# Analytics with DuckDB + Parquet
uv sync --extra analytics

# Terminal UI
uv sync --extra tui

# MCP server for Claude Desktop (coming soon)
uv sync --extra mcp

# Web API server (coming soon)
uv sync --extra serve

# Everything
uv sync --extra all
```

## Configuration

Config lives at `~/.ricoeur/config.toml`:

```toml
[general]
home = "~/.ricoeur"
default_language = "en"

[embeddings]
model = "st:paraphrase-multilingual-mpnet-base-v2"
batch_size = 64
device = "auto"

# Topics and summarize config — coming soon
# [topics]
# min_cluster_size = 15
# n_topics = "auto"
#
# [summarize]
# enabled = false
# model = "ollama:llama3.2"
```

Override the data directory with the `RICOEUR_HOME` environment variable.

## Architecture

```
~/.ricoeur/
├── config.toml          # Configuration
├── ricoeur.db           # SQLite database (FTS5 search)
├── analytics/           # Parquet files for DuckDB
├── embeddings/          # Sentence-transformer vectors
├── models/              # Saved BERTopic models (coming soon)
└── attachments/         # Extracted files (coming soon)
```

## License

MIT
