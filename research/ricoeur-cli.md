# ricoeur

**A local-first archive, search, and intelligence engine for your LLM conversation history.**

*Named after Paul Ricoeur, whose work on narrative identity argued that we understand ourselves through the stories we construct from our lived experience. ricoeur reconstructs the narrative of your intellectual life from thousands of AI conversations.*

---

## Installation

```bash
# macOS / Linux
curl -fsSL https://ricoeur.dev/install.sh | bash

# From source (requires Go 1.25+, CGO for SQLite/DuckDB)
git clone https://github.com/yourusername/ricoeur.git
cd ricoeur
make install

# Python (alternative, for users who want to extend with notebooks)
pip install ricoeur
```

---

## Quickstart

```bash
# Initialize the database
ricoeur init

# Import your ChatGPT export
ricoeur import chatgpt ~/Downloads/chatgpt-export/conversations.json

# Import your Claude export
ricoeur import claude ~/Downloads/claude-export/conversations.json

# See what you've got
ricoeur stats

# Build topic model and embeddings
ricoeur index

# Search your history
ricoeur search "thermal simulation CFD"

# Launch the TUI
ricoeur tui

# Start the MCP server
ricoeur mcp
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Import Sources                                             │
│  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ ChatGPT  │  │ Claude  │  │  Gemini  │  │ Custom JSON │ │
│  └────┬─────┘  └────┬────┘  └────┬─────┘  └──────┬──────┘ │
│       └──────────────┴───────────┴───────────────┘         │
│                          │                                  │
│  Storage Layer           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SQLite (system of record)                          │   │
│  │  ├── conversations   (id, title, platform, model,   │   │
│  │  │                    created_at, language, topic)   │   │
│  │  ├── messages        (id, conv_id, role, content,   │   │
│  │  │                    timestamp, content_type)       │   │
│  │  ├── code_blocks     (id, msg_id, language, code)   │   │
│  │  ├── attachments     (id, conv_id, type, path)      │   │
│  │  └── summaries       (conv_id, summary, model_used) │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌──────────────────────┐  ┌────────────────────────────┐  │
│  │  Parquet (analytics) │  │  Embeddings (sqlite-vec    │  │
│  │  DuckDB-powered      │  │  or .npy sidecar files)    │  │
│  │  aggregate queries   │  │  semantic search + UMAP    │  │
│  └──────────────────────┘  └────────────────────────────┘  │
│                                                             │
│  Intelligence Layer                                         │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  │
│  │ FTS5     │  │ BERTopic  │  │ Ollama   │  │ Language │  │
│  │ fulltext │  │ topics +  │  │ summaries│  │ detect   │  │
│  │ search   │  │ hierarchy │  │ + labels │  │ (fasttext)│  │
│  └──────────┘  └───────────┘  └──────────┘  └──────────┘  │
│                                                             │
│  Interface Layer                                            │
│  ┌───────┐  ┌───────┐  ┌─────────┐  ┌──────────────────┐  │
│  │  CLI  │  │  TUI  │  │   MCP   │  │  Web (optional)  │  │
│  │       │  │       │  │  Server │  │  Streamlit / API  │  │
│  └───────┘  └───────┘  └─────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Commands

### `ricoeur init`

Initialize a new ricoeur database.

```bash
ricoeur init                          # default: ~/.ricoeur/
ricoeur init --home /mnt/data/ricoeur # custom location
```

Creates the directory structure:

```
~/.ricoeur/
├── config.toml
├── ricoeur.db          # SQLite (system of record)
├── analytics/          # Parquet files for DuckDB
├── embeddings/         # numpy arrays, model metadata
├── models/             # saved BERTopic models
└── attachments/        # extracted images, files (content-addressed)
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--home` | `~/.ricoeur` | Data directory (also: `RICOEUR_HOME` env var) |

---

### `ricoeur import <platform> <path>`

Import conversations from an LLM platform export.

```bash
# ChatGPT (from Settings → Data Controls → Export Data)
ricoeur import chatgpt conversations.json
ricoeur import chatgpt ~/Downloads/chatgpt-2026-02-14.zip  # accepts zip too

# Claude (from Settings → Export my data)
ricoeur import claude conversations.json

# Gemini (Google Takeout)
ricoeur import gemini takeout-data.json

# Generic JSON (custom schema, provide a mapping file)
ricoeur import custom data.json --mapping my_mapping.toml

# Re-import (updates existing, adds new, never deletes)
ricoeur import chatgpt conversations.json --update
```

**What import does:**

1. Parses platform-specific JSON structure (ChatGPT tree-walking, Claude flat arrays)
2. Normalizes into the unified schema (conversations + messages)
3. Detects language per conversation (fasttext `lid.176.bin`)
4. Extracts code blocks, tags by language
5. Catalogs attachments, DALL-E generations, uploaded files
6. Deduplicates by conversation ID (safe to re-import)

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--update` | `false` | Update existing conversations if reimporting |
| `--mapping` | — | TOML file mapping custom JSON fields to ricoeur schema |
| `--since` | — | Only import conversations after this date (ISO format) |
| `--dry-run` | `false` | Parse and validate without writing to database |
| `--extract-attachments` | `true` | Copy/link attachment files to ricoeur store |

**Output:**

```
Importing from ChatGPT...
  Parsed:       3,756 conversations (53,808 messages)
  New:          3,756
  Updated:      0
  Skipped:      0
  Languages:    en: 2,891 | fr: 724 | mixed: 141
  Code blocks:  7,794 extracted
  Attachments:  525 cataloged
  Duration:     4.2s
```

---

### `ricoeur index`

Build or rebuild the intelligence layers: embeddings, topic model, and analytics cache.

```bash
# Full index (embeddings + topics + analytics)
ricoeur index

# Only rebuild specific layers
ricoeur index --embeddings
ricoeur index --topics
ricoeur index --analytics

# Use Ollama for embeddings instead of sentence-transformers
ricoeur index --embeddings --embed-model ollama:nomic-embed-text

# Customize topic model
ricoeur index --topics --n-topics auto --min-cluster-size 25

# Generate summaries using a local LLM
ricoeur index --summarize --llm ollama:llama3.2

# Generate LLM-powered topic labels
ricoeur index --topics --label-model ollama:llama3.2
```

**Sub-operations:**

| Operation | What it does | Time estimate (3,756 convs) |
|-----------|-------------|---------------------------|
| `--embeddings` | Encode all conversations with sentence-transformers or Ollama | 5–30 min |
| `--topics` | Run BERTopic (UMAP → HDBSCAN → c-TF-IDF) | 1–3 min |
| `--analytics` | Export Parquet metadata for DuckDB | <10 sec |
| `--summarize` | Generate 1-sentence summaries via local LLM | 1–3 hours |
| `--label-model` | Use LLM to name topics (runs after --topics) | 2–5 min |

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--embed-model` | `st:paraphrase-multilingual-mpnet-base-v2` | Embedding model (`st:` for sentence-transformers, `ollama:` for Ollama) |
| `--n-topics` | `auto` | Number of topics (`auto` lets HDBSCAN decide) |
| `--min-cluster-size` | `15` | Minimum conversations per topic |
| `--label-model` | — | LLM for generating human-readable topic labels |
| `--summarize` | `false` | Generate per-conversation summaries |
| `--llm` | `ollama:llama3.2` | LLM to use for summaries |
| `--force` | `false` | Reindex everything, even if unchanged |
| `--batch-size` | `64` | Batch size for embedding generation |
| `--device` | `auto` | `cpu`, `cuda`, `mps`, or `auto` |

**Output:**

```
Indexing...
  Embeddings:   3,756 conversations encoded (model: paraphrase-multilingual-mpnet-base-v2)
                768 dimensions, saved to ~/.ricoeur/embeddings/
  Topics:       42 topics detected, 127 outliers reassigned
                Top topics: MATLAB debugging (284), Python API integration (201),
                            Competitive analysis (178), MCP server setup (134)...
  Analytics:    Parquet cache rebuilt (3 files, 2.1 MB)
  Duration:     8m 32s
```

---

### `ricoeur search <query>`

Search across all conversations. Combines full-text search (FTS5) and semantic search (embeddings).

```bash
# Basic full-text search
ricoeur search "streamlit dashboard"

# Semantic search (finds conceptually related conversations, not just keyword matches)
ricoeur search "how to visualize data interactively" --semantic

# Hybrid search (combines keyword + semantic, default when index exists)
ricoeur search "thermal simulation"

# Filter by platform, language, date, topic, model
ricoeur search "error fix" --platform chatgpt --lang en
ricoeur search "stratégie marketing" --lang fr --since 2025-01-01
ricoeur search "debugging" --model gpt-5 --until 2025-06-30
ricoeur search "competitive analysis" --topic 7

# Search only in code blocks
ricoeur search "import pandas" --code

# Search within a specific conversation
ricoeur search "UMAP parameters" --conversation abc123

# Output formats
ricoeur search "MCP" --format table           # default: compact table
ricoeur search "MCP" --format json            # machine-readable
ricoeur search "MCP" --format full            # show message excerpts
ricoeur search "MCP" --format ids             # just conversation IDs
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--semantic` | `false` | Force semantic-only search |
| `--keyword` | `false` | Force keyword-only search (FTS5) |
| `--platform` | all | Filter: `chatgpt`, `claude`, `gemini` |
| `--lang` | all | Filter: `en`, `fr`, `mixed`, etc. |
| `--since` | — | Only conversations after this date |
| `--until` | — | Only conversations before this date |
| `--model` | all | Filter by LLM model slug |
| `--topic` | all | Filter by topic ID or name |
| `--code` | `false` | Search only in extracted code blocks |
| `--role` | all | Filter by message role: `user`, `assistant` |
| `--limit` | `20` | Max results |
| `--format` | `table` | Output: `table`, `json`, `full`, `ids` |
| `--conversation` | — | Search within a specific conversation |

**Output (default table):**

```
Found 14 results for "thermal simulation" (hybrid search)

 #  Score  Date        Platform  Lang  Topic                  Title
 1  0.94   2025-11-03  chatgpt   en    Simulation Software    COMSOL vs ANSYS thermal
 2  0.91   2025-10-28  chatgpt   en    Simulation Software    Thermal analysis MATLAB PDE
 3  0.87   2025-12-15  claude    en    Competitive Analysis   Thermal simulation market map
 4  0.83   2025-09-12  chatgpt   en    CFD Workflows          Heat transfer CFD setup
 ...

Use `ricoeur show <id>` to read a conversation.
```

---

### `ricoeur show <conversation_id>`

Display a full conversation with metadata and formatting.

```bash
ricoeur show abc123def
ricoeur show abc123def --messages user       # only show user messages
ricoeur show abc123def --messages assistant   # only show assistant messages
ricoeur show abc123def --summary              # show summary + metadata only
ricoeur show abc123def --code                 # only show code blocks
ricoeur show abc123def --format md            # output as markdown
ricoeur show abc123def --format json          # raw JSON
```

**Output:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 COMSOL vs ANSYS thermal analysis comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Platform:  ChatGPT          Model:   gpt-5
 Date:      2025-11-03       Messages: 18
 Language:  English           Topic:   Simulation Software (#7)
 Summary:   Compared thermal simulation capabilities of COMSOL
            Multiphysics and ANSYS for automotive heat exchanger
            design, focusing on meshing, solver options, and
            MATLAB integration workflows.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 You (10:34 AM)
 I need to compare COMSOL and ANSYS for thermal simulation
 of a heat exchanger. What are the key differences in their
 meshing approaches?

 Assistant (10:34 AM)  [gpt-5]
 For heat exchanger thermal simulation, COMSOL and ANSYS
 take notably different approaches to meshing...
 ...
```

---

### `ricoeur topics`

Explore and manage the topic model.

```bash
# List all topics with stats
ricoeur topics list

# Show details for a specific topic
ricoeur topics show 7

# Show topic hierarchy (tree view)
ricoeur topics tree

# Show how topics evolved over time
ricoeur topics timeline

# Show topics for a date range
ricoeur topics timeline --since 2025-01-01 --until 2025-12-31

# Merge two topics manually
ricoeur topics merge 12 15 --label "Python Development"

# Rename a topic
ricoeur topics rename 7 "Engineering Simulation"

# List conversations in a topic
ricoeur topics conversations 7

# Compare topics across platforms
ricoeur topics compare --by platform

# Export topic model
ricoeur topics export topics.json
```

**`ricoeur topics list` output:**

```
42 topics across 3,756 conversations

 ID  Label                        Count   Top Keywords                    Trend
  0  MATLAB Debugging               284   matlab, error, fix, function    ━━━━━━━▶
  1  Python API Integration         201   python, api, request, json      ━━━━━▶
  2  Competitive Analysis           178   market, competitor, strategy    ━━━━━▶
  3  MCP Server Development         134   mcp, server, claude, tool       ━━━▲▲▲
  4  Streamlit Applications         112   streamlit, app, dashboard       ━━━━━━▶
  5  Content Writing (FR)            98   blog, article, contenu, rédac   ━━━━▶
  6  Git & DevOps                    95   git, github, ssh, deploy        ━━━━━━▶
  7  Engineering Simulation          87   simulation, thermal, cfd, fem   ━━━▶
  8  AI Strategy & Positioning       82   strategy, positioning, market   ━━━━▶
  9  Philosophy & Ideas              74   philosophy, existential, idea   ━━━━━━▶
 ...

 ▶ = stable   ▲ = growing   ▼ = declining
```

**`ricoeur topics tree` output:**

```
All Topics (3,756 conversations)
├── Technical (1,892)
│   ├── MATLAB & Simulink (412)
│   │   ├── MATLAB Debugging (284)
│   │   ├── Simulink Model-Based Design (78)
│   │   └── MATLAB Toolbox Usage (50)
│   ├── Python & Web Dev (498)
│   │   ├── Python API Integration (201)
│   │   ├── Streamlit Applications (112)
│   │   ├── Git & DevOps (95)
│   │   └── HTML/CSS/Frontend (90)
│   ├── AI & ML (367)
│   │   ├── MCP Server Development (134)
│   │   ├── LLM Integration (98)
│   │   ├── Embeddings & NLP (72)
│   │   └── Computer Vision (63)
│   └── Engineering (298)
│       ├── Engineering Simulation (87)
│       ├── CFD & Thermal (68)
│       └── ...
├── Strategic (648)
│   ├── Competitive Analysis (178)
│   ├── AI Strategy & Positioning (82)
│   ├── Market Intelligence (64)
│   └── ...
├── Content & Creative (512)
│   ├── Content Writing (FR) (98)
│   ├── Blog & Article Drafting (87)
│   ├── Video & Media (76)
│   └── ...
└── Personal & Misc (704)
    ├── Philosophy & Ideas (74)
    ├── Weather & Location (78)
    └── ...
```

---

### `ricoeur stats`

Analytics dashboard in the terminal. Powered by DuckDB over Parquet.

```bash
# Overview
ricoeur stats

# Detailed breakdowns
ricoeur stats --monthly                  # conversations per month
ricoeur stats --models                   # usage by model over time
ricoeur stats --platforms                # ChatGPT vs Claude vs Gemini
ricoeur stats --languages                # language distribution
ricoeur stats --topics                   # topic distribution + trends
ricoeur stats --code                     # programming languages used
ricoeur stats --hours                    # activity by hour of day

# Filtered stats
ricoeur stats --since 2025-01-01 --until 2025-12-31
ricoeur stats --platform chatgpt
ricoeur stats --lang fr

# Export as JSON or CSV
ricoeur stats --monthly --format csv > monthly.csv
ricoeur stats --format json > stats.json
```

**`ricoeur stats` output:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ricoeur — your narrative in numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Conversations    3,756      Messages       53,808
 Platforms        2          Models used    24
 Date range       Dec 2022 — Feb 2026 (38 months)
 Languages        en: 77%  fr: 19%  mixed: 4%
 Topics           42 detected
 Your words       1,353,226  (~5 novels worth)

 ┌─────────── Conversations per month ──────────┐
 │                                    ▓▓        │
 │                                  ▓▓▓▓  ▓▓   │
 │                              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
 │                          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
 │      ▓▓  ▓▓      ▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
 │ 2023        2024           2025        2026  │
 └──────────────────────────────────────────────┘

 Top models          Top topics (by count)
 gpt-4o     1,053    MATLAB Debugging      284
 gpt-5        398    Python API            201
 gpt-5-2      354    Competitive Analysis  178
 o4-mini-h    237    MCP Development       134
 o3-mini-h    186    Streamlit Apps        112
```

---

### `ricoeur mcp`

Start the MCP server. This is the primary interface for AI-powered exploration.

```bash
# Start MCP server (stdio mode, for Claude Desktop / agent integration)
ricoeur mcp

# Start on a specific transport
ricoeur mcp --transport stdio           # default, for Claude Desktop
ricoeur mcp --transport tcp --port 3100 # for network agents

# With verbose logging
ricoeur mcp --verbose
```

**MCP Tools exposed:**

| Tool | Description |
|------|-------------|
| `search` | Full-text + semantic search across all conversations |
| `search_code` | Search extracted code blocks by language and content |
| `get_conversation` | Retrieve full conversation by ID |
| `get_summary` | Get conversation summary and metadata |
| `list_topics` | List all topics with counts and keywords |
| `get_topic` | Get topic details, representative conversations |
| `topic_timeline` | Topic frequency over time |
| `stats` | Aggregate statistics with optional filters |
| `find_similar` | Find conversations similar to a given one |
| `compare_periods` | Compare topic distributions between two date ranges |
| `list_models` | List all models used with conversation counts |
| `list_code_languages` | List programming languages found in code blocks |

**Claude Desktop configuration (`claude_desktop_config.json`):**

```json
{
  "mcpServers": {
    "ricoeur": {
      "command": "ricoeur",
      "args": ["mcp"],
      "env": {
        "RICOEUR_HOME": "/Users/yann/.ricoeur"
      }
    }
  }
}
```

**Example agent interactions once connected:**

```
You:    "When did I first start working with MCP servers?"
Claude: [calls ricoeur.search("MCP server", sort="date_asc", limit=5)]
        "Your earliest MCP conversation was on December 8, 2024:
         'Claude Desktop MCP Servers' — you were setting up MCP
         server configuration for Claude Desktop..."

You:    "What were my main topics last quarter?"
Claude: [calls ricoeur.compare_periods(
           period_a="2025-10-01/2025-12-31",
           period_b="2025-07-01/2025-09-30")]
        "In Q4 2025, your top emerging topics were..."

You:    "Find everything I discussed about COMSOL"
Claude: [calls ricoeur.search("COMSOL", format="full")]
        "I found 12 conversations mentioning COMSOL..."
```

---

### `ricoeur tui`

Launch the interactive terminal UI. Drill-down by topic, date, platform, model, language.

```bash
ricoeur tui
```

**Key bindings:**

| Key | Action |
|-----|--------|
| `/` | Open search |
| `t` | Topic browser |
| `s` | Stats dashboard |
| `h` | Topic hierarchy (tree) |
| `m` | Timeline view (monthly) |
| `Enter` | Open selected conversation |
| `f` | Toggle filters panel |
| `q` | Quit |
| `?` | Help |

---

### `ricoeur serve`

Start a local web server (REST API + optional Streamlit dashboard).

```bash
# API only
ricoeur serve --port 8080

# With Streamlit dashboard
ricoeur serve --port 8080 --dashboard

# API + dashboard, bind to all interfaces
ricoeur serve --port 8080 --dashboard --host 0.0.0.0
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | GET | Search with query params |
| `/api/conversations/:id` | GET | Get conversation |
| `/api/conversations/:id/messages` | GET | Get messages |
| `/api/topics` | GET | List topics |
| `/api/topics/:id` | GET | Topic detail |
| `/api/topics/timeline` | GET | Topic evolution data |
| `/api/stats` | GET | Aggregate statistics |
| `/api/embeddings/umap` | GET | 2D UMAP projection (for scatter plots) |
| `/api/similar/:id` | GET | Find similar conversations |

---

### `ricoeur export`

Export data in various formats for external use.

```bash
# Export all conversations as markdown files (one per conversation)
ricoeur export markdown ./output/

# Export as structured JSON
ricoeur export json conversations.json

# Export as CSV (flat table, one row per conversation)
ricoeur export csv conversations.csv

# Export embeddings as numpy array
ricoeur export embeddings embeddings.npy

# Export topic model
ricoeur export topics topics.json

# Export for Nomic Atlas
ricoeur export atlas atlas_upload.json

# Filtered exports
ricoeur export markdown ./matlab-convos/ --topic 0 --lang en
ricoeur export csv french.csv --lang fr --since 2025-01-01
```

---

### `ricoeur config`

View and edit configuration.

```bash
# Show current config
ricoeur config show

# Set defaults
ricoeur config set embed_model "ollama:nomic-embed-text"
ricoeur config set llm "ollama:llama3.2"
ricoeur config set min_cluster_size 25
ricoeur config set default_language "en"
```

**`~/.ricoeur/config.toml`:**

```toml
[general]
home = "~/.ricoeur"
default_language = "en"

[embeddings]
model = "st:paraphrase-multilingual-mpnet-base-v2"
# model = "ollama:nomic-embed-text"    # alternative
batch_size = 64
device = "auto"                         # auto, cpu, cuda, mps

[topics]
min_cluster_size = 15
n_topics = "auto"
label_model = ""                        # e.g. "ollama:llama3.2"

[summarize]
enabled = false
model = "ollama:llama3.2"
max_input_tokens = 2000

[mcp]
transport = "stdio"
port = 3100                             # for TCP transport

[serve]
port = 8080
host = "127.0.0.1"
dashboard = true
```

---

## Full command tree

```
ricoeur
├── init                    Initialize database
├── import                  Import conversation data
│   ├── chatgpt             From ChatGPT export
│   ├── claude              From Claude export
│   ├── gemini              From Gemini/Takeout export
│   └── custom              From custom JSON with mapping
├── index                   Build intelligence layers
│   ├── --embeddings        Generate embeddings
│   ├── --topics            Run topic model
│   ├── --analytics         Rebuild Parquet cache
│   └── --summarize         Generate LLM summaries
├── search <query>          Search conversations
├── show <id>               Display a conversation
├── topics                  Topic model operations
│   ├── list                List all topics
│   ├── show <id>           Topic details
│   ├── tree                Hierarchical view
│   ├── timeline            Topic evolution over time
│   ├── conversations <id>  List conversations in topic
│   ├── merge <a> <b>       Merge two topics
│   ├── rename <id> <name>  Rename a topic
│   ├── compare             Compare across dimensions
│   └── export              Export topic model
├── stats                   Analytics dashboard
├── mcp                     Start MCP server
├── tui                     Launch terminal UI
├── serve                   Start web server + API
├── export                  Export data
│   ├── markdown            One .md file per conversation
│   ├── json                Structured JSON
│   ├── csv                 Flat CSV table
│   ├── embeddings          Numpy array
│   ├── topics              Topic model JSON
│   └── atlas               Nomic Atlas format
├── config                  View/edit configuration
│   ├── show                Print current config
│   └── set <key> <value>   Set a config value
└── version                 Print version info
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `RICOEUR_HOME` | Override data directory (default: `~/.ricoeur`) |
| `RICOEUR_EMBED_MODEL` | Override embedding model |
| `RICOEUR_LLM` | Override LLM for summaries/labels |
| `OLLAMA_HOST` | Ollama server address (default: `http://localhost:11434`) |

---

## Design principles

1. **Local-first.** Your conversations never leave your machine. No cloud, no telemetry, no API keys required for core functionality.

2. **Import once, query forever.** The initial import is the only slow step. Everything after runs against local SQLite, Parquet, and embeddings.

3. **MCP as the primary AI interface.** The MCP server is not an afterthought — it's the main way you interact with ricoeur through AI assistants. Your past becomes context for your present.

4. **Progressive intelligence.** Works immediately with just import + FTS5 search. Add embeddings for semantic search. Add topics for clustering. Add Ollama for summaries. Each layer is optional and incremental.

5. **Multi-platform by design.** ChatGPT, Claude, Gemini, and custom sources share one unified schema. Your narrative isn't platform-specific.

6. **Narrative, not just data.** Following Ricoeur's philosophy: the goal isn't to store conversations, it's to help you understand the story they tell about your intellectual evolution.
