# Embedding Model Comparison

Tested on 3,722 conversations (3,200 English, 312 French, 223 unknown).

## Footprint

| | **paraphrase-multilingual-mpnet-base-v2** | **all-MiniLM-L6-v2** |
|---|---|---|
| Parameters | 278M | 22M |
| Model on disk | 1.0 GB | 103 MB |
| Model in RAM (fp32) | 1,061 MB | 87 MB |
| Embedding dim | 768 | 384 |
| Embeddings file (3,722 convs) | 11 MB | 5.5 MB |
| Max seq length | 128 tokens | 256 tokens |
| Languages | 50+ languages | English only |

MiniLM is ~10x lighter across the board.

## Search Quality

### Query: "how to containerize applications"

Keyword search returns only **3 results** for this query. Both models find 20.

#### Multilingual (768d)

| # | Score | Title |
|---|-------|-------|
| 1 | 0.5515 | Free docker deployment options |
| 2 | 0.5152 | Secure Clawdbot Setup |
| 3 | 0.5081 | Docker noVNC Setup |
| 4 | 0.5014 | HTML upload and serve |
| 5 | 0.4782 | Wasm vs Container Comparison |
| 6 | 0.4687 | MATLAB vs Python |
| 7 | 0.4613 | Deploy MATLAB on AWS |
| 8 | 0.4573 | Flask-Markdown App Template |
| 9 | 0.4562 | Project ZIP Creation |
| 10 | 0.4555 | HTML to Markdown Conversion |

#### MiniLM (384d)

| # | Score | Title |
|---|-------|-------|
| 1 | 0.4665 | Wasm vs Container Comparison |
| 2 | 0.4029 | Différence parties Dock Apple |
| 3 | 0.3795 | Run OpenFOAM in container |
| 4 | 0.3663 | Docker Ephemeral File System |
| 5 | 0.3649 | Deploying Claude Artifacts Apps |
| 6 | 0.3639 | ADD vs COPY in Dockerfile |
| 7 | 0.3497 | Free docker deployment options |
| 8 | 0.3461 | Docker compose on Render |
| 9 | 0.3433 | Secure Clawdbot Setup |
| 10 | 0.3426 | Docker run -t -i |

**Observations:**
- Both models surface Docker/container content effectively
- Multilingual has higher confidence scores and more precise top-5
- MiniLM has a false positive at #2: "Différence parties Dock Apple" (macOS Dock, not Docker) — likely a lexical false match on "Dock"
- MiniLM surfaces more Docker-specific titles (ADD vs COPY, docker run flags) in the top 10

### Query: "deployment strategies"

#### Multilingual

Top hits: 1-Year Plan for PMM, Development Areas, Warzone Perk Loadouts (!), Work Procedure Implementation, Lunch Conversation Strategy

#### MiniLM

Top hits: Branching strategies (0.44), Deploying Claude Artifacts Apps (0.44), Productizing FDE (0.41), MCP server ideas (0.39)

**Observations:**
- MiniLM is more precise here — "Branching strategies" is genuinely about deployment
- Multilingual picks up broader "strategy" concepts but includes noise (Warzone loadouts)
- For English queries, MiniLM can match or exceed multilingual quality

### Query: "python" --lang fr (hybrid mode)

Both models correctly surface French Python conversations. Multilingual ranks "Intégration Zapier avec Python" and "Extraire historique Python" at the top. MiniLM leads with "Extraire historique Python" and "Méthodes API Excel".

## Recommendation

| Use case | Model |
|----------|-------|
| Multilingual archive (mixed en/fr) | `st:paraphrase-multilingual-mpnet-base-v2` |
| English-only, memory-constrained | `st:all-MiniLM-L6-v2` |
| Local/offline with Ollama | `ollama:nomic-embed-text` |

For this archive (8% French), the multilingual model is the right default. The 1 GB disk cost is paid once and cached at `~/.cache/huggingface/`.

## How to switch

```bash
# Use MiniLM
ricoeur config set embeddings.model st:all-MiniLM-L6-v2
ricoeur index --embeddings --force

# Use Ollama (requires running ollama server)
ricoeur config set embeddings.model ollama:nomic-embed-text
ricoeur index --embeddings --force

# Back to default
ricoeur config set embeddings.model st:paraphrase-multilingual-mpnet-base-v2
ricoeur index --embeddings --force
```
