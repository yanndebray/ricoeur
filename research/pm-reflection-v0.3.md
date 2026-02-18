# Product Manager Reflection: ricoeur v0.3.3

## What we've shipped

ricoeur has gone from "zero" to a functional local-first CLI tool in remarkably few iterations. Here's the feature map:

| Command | Status | Notes |
|---------|--------|-------|
| `init` | Shipped | SQLite + FTS5, config.toml, directory structure |
| `import chatgpt` | Shipped | Tree-walking parser, dedup, ~3,700 conversations imported |
| `search` | Shipped (v0.3.0) | Three modes: keyword (FTS5/BM25), semantic (cosine sim), hybrid (RRF). Filters: platform, lang, date, model, topic, code, role |
| `show` | Shipped (v0.3.1) | Full conversation viewer, partial ID prefix matching |
| `stats` | Shipped | Basic counts — conversations, messages, models, languages, monthly breakdown |
| `index` | Partial | Embeddings + language detection work. Topics and analytics cache are not yet wired |
| `config` | Shipped | Show + set, TOML-backed |
| Website | Shipped | ricoeur.cc on Netlify |
| PyPI | Shipped | `pip install ricoeur`, GitHub Actions publish workflow |

**Key technical wins:**
- Hybrid search with RRF fusion — the demo of "deployment strategies" (3 FTS results vs 20 hybrid) is a compelling proof of the intelligence layer concept
- Multilingual embedding model chosen after rigorous comparison (documented in `research/embedding-models.md`)
- Progressive dependency model — core is just click + rich + sqlite, extras opt-in via `[embeddings]`, `[topics]`, etc.

## Where we are vs. the vision

The research spec (`research/ricoeur-cli.md`) describes a 12-command, multi-platform, multi-interface system. Here's the honest gap:

| Vision layer | Built | Gap |
|---|---|---|
| **Import** | ChatGPT only | Claude, Gemini, custom JSON imports not started |
| **Intelligence** | FTS5 + embeddings + langdetect | BERTopic (#4), LLM summaries, DuckDB analytics cache all missing |
| **Search** | 3 modes, 8 filters | `--conversation`, `--format json/full/ids` not implemented |
| **Interfaces** | CLI only | MCP (#5), TUI (#6), web serve — all unstarted |
| **Export** | None | markdown/json/csv/atlas export not started |

We're roughly at **35-40% of the full vision**, heavily weighted toward the storage + search core.

## What to build next — prioritized

### Tier 1: High impact, unlocks the narrative (next 2 releases)

1. **#5 MCP server** — This is the soul of the product. The research spec says "MCP is the primary AI interface, not an afterthought." Once Claude Desktop can call `ricoeur.search()` and `ricoeur.get_conversation()`, the archive becomes living context for daily work. The dependency (`mcp>=1.0`) is already declared. This turns ricoeur from a CLI novelty into a genuine productivity tool.

2. **#4 BERTopic topics** — Topic modeling is the second intelligence layer. It transforms the archive from "searchable conversations" to "a map of your intellectual life." The NLP sprint (weeks 2-4) already proved the pipeline works. The `topics` command tree (list, show, tree, timeline) is fully designed. This also powers better search (filter by topic) and gives stats much more depth.

### Tier 2: UX polish, broader reach

3. **#6 TUI** — A Textual-based interactive browser. This is the "wow" demo moment: type `ricoeur tui`, see your topics as a tree, drill into conversations. But it depends on topics being functional first, so sequence it after #4.

4. **#7 litellm support** — Replacing the hard-coded ollama backend with litellm opens up OpenAI, Anthropic, and dozens of providers for embeddings and summaries. Important for users who don't run Ollama locally, but the current sentence-transformers default already works well offline.

5. **#2 index command completion** — The index command works for embeddings and langdetect, but `--topics` and `--analytics` are stubs. Completing these is a prerequisite for #4 (topics) and for richer stats.

### Tier 3: Platform expansion

6. **Claude/Gemini import** — Multi-platform is a design principle, but the user's archive is 100% ChatGPT today. This matters more when shipping to other users.

7. **Export command** — Useful for interop (feed conversations to other tools, create markdown archives), but not urgent for single-user workflow.

8. **#8 Website UI** — Cosmetic alignment with philosophers.cc. Low priority vs. feature work.

## Recommended roadmap

| Release | Focus | Key deliverables |
|---------|-------|-------------------|
| **v0.4.0** | MCP server | `ricoeur mcp` with search, get_conversation, stats, list_topics tools. Claude Desktop integration. |
| **v0.5.0** | Topics | Complete `ricoeur index --topics`, add `ricoeur topics list/show/tree`. Wire topic filter into search. |
| **v0.6.0** | TUI | `ricoeur tui` with search, topic browser, conversation viewer |
| **v0.7.0** | Multi-provider | litellm support for embeddings + summaries, Claude import |

## Risks and considerations

- **Scope creep** — The research spec is ambitious (12 commands, 4 interfaces). The biggest risk is building breadth before depth. MCP alone, done well, could be the entire v1.0 story.
- **Single-user assumption** — Everything is built for one user's archive. If this ships to others, we need better onboarding, error messages, and documentation for the `init → import → index → search` pipeline.
- **Embedding model size** — The 1GB multilingual model is the right default for this archive (8% French), but it's a friction point for first-time users. Consider lazy download with a progress bar, or defaulting to MiniLM with a prompt to upgrade.
- **Test coverage** — We have 31 tests for search, but nothing for import, index, show, stats, or config. Before the MCP server (which becomes an external interface), we need broader coverage.

## The narrative pitch

ricoeur already answers the question: *"What did I talk about with AI?"* With search in three modes and partial-ID drill-down, it's genuinely useful today.

The next milestone should answer: *"What can AI learn from my past conversations?"* That's MCP. When Claude can search your archive and say "You asked about this exact pattern six months ago — here's what you decided," ricoeur fulfills its Ricoeurian promise: your past conversations become narrative identity, not dead data.
