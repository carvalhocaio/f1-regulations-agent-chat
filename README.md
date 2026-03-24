# F1 Regulations Agent Chat

AI assistant for Formula 1 that combines:
- Official **FIA 2026 regulations** (hybrid RAG over PDFs)
- Historical **F1 World Championship data (1950-2024)** in SQLite
- **Current and time-sensitive** information via Google Search

Built with [Google ADK](https://google.github.io/adk-docs/) and powered by Gemini.

> This project is based on [f1-regulations-agent](https://github.com/carvalhocaio/f1-regulations-agent), the original CLI version.

## Core Capabilities

**Tools:**
- `search_regulations` — hybrid retrieval (FAISS semantic + BM25 keyword) over FIA 2026 Sections A-F
- `query_f1_history_template` — pre-built SQL templates for common F1 queries (champions, career stats, records, standings, head-to-head)
- `query_f1_history` — read-only SQL access to historical F1 data
- `google_search_agent` — live web retrieval for current season/news
- `search` (compatibility alias) — guided fallback if the model hallucinates a generic tool name

**Intelligence layer:**
- **Model routing** — simple queries go to Flash (or fine-tuned Flash), complex ones stay on Pro
- **Semantic cache** — near-instant responses for repeated/similar questions (cosine similarity > 0.92)
- **Session corrections** — detects when users correct the agent (PT/EN) and avoids repeating mistakes
- **Temporal reasoning** — automatically splits questions: `1950-2024` via SQLite, `2025+` via web search

## How It Works

```text
User question
    |
    v
[check_cache] -----> Cache HIT? Return cached answer (<200ms)
    |
[inject_corrections] --> Append session corrections to prompt
    |
[route_model] -----> Simple? -> Flash/Tuned | Complex? -> Pro
    |
    v
ADK Agent (Gemini)
    |
    |-- Regulations -----------> search_regulations(query)
    |                             -> FAISS + BM25 hybrid search
    |
    |-- Historical/stats ------> query_f1_history_template(name, params)
    |                             -> Pre-built SQL templates
    |
    |-- Free-form SQL ---------> query_f1_history(sql_query)
    |                             -> SQLite (1950-2024, SELECT only)
    |
    |-- Current/live ----------> google_search_agent(request)
    |                             -> Web results
    |
    v
[detect_corrections] --> Store if user corrected the agent
    |
[store_cache] --------> Cache the answer for future reuse
    |
    v
Unified answer + sources
```

## Stack

- [Google ADK](https://google.github.io/adk-docs/) — agent framework and local web UI
- [Gemini 2.5 Pro / Flash](https://deepmind.google/technologies/gemini/) — LLM with dynamic routing
- [LangChain](https://python.langchain.com/) + [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF ingestion and chunking
- [FAISS](https://github.com/facebookresearch/faiss) + [BM25](https://github.com/dorianbrown/rank_bm25) — hybrid vector + keyword search with reciprocal rank fusion
- [SQLite](https://www.sqlite.org/) — historical F1 database with pre-built query templates
- Vertex AI Agent Engine — production deployment target
- Vertex AI SFT — supervised fine-tuning pipeline for domain-specific Flash model
- Terraform + GitHub Actions — infrastructure and CI/CD

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/carvalhocaio/f1-regulations-agent-chat
cd f1-regulations-agent-chat
uv sync

# 3. Configure .env
echo 'GOOGLE_API_KEY=your-key-here' > .env

# 4. Add source data to docs/ (FIA PDFs + Kaggle CSVs)

# 5. Build artifacts
uv run build_index.py

# 6. Run locally
make run
```

For detailed setup instructions, see [DEVELOPMENT.md](./DEVELOPMENT.md).

## Example Questions

**Regulations:**
- "What is the maximum power unit energy deployment per lap?"
- "What are the dimensions allowed for the front wing?"
- "What is the cost cap for F1 teams?"

**Historical:**
- "How many wins did Ayrton Senna have?"
- "Which constructor has the most championships?"
- "Compare Hamilton and Schumacher race wins."

**Current/live:**
- "What is the race calendar for this season?"
- "Who leads the championship right now?"

**Cross-source:**
- "Compare Schumacher's 2004 dominance with 2026 regulation changes."
- "Show the last 10 drivers' champions."

## SQL Tool Constraints

`query_f1_history` is intentionally restricted:
- Only `SELECT` queries are allowed
- Write operations are blocked (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.)
- Results are capped at `100` rows

## Project Structure

```text
f1-regulations-agent-chat/
├── f1_agent/
│   ├── agent.py               # ADK agent definition, tools, and callbacks
│   ├── callbacks.py            # Model routing, semantic cache, session corrections
│   ├── cache.py                # SemanticCache (FAISS + SQLite, TTL-based)
│   ├── tools.py                # Agent tools (regulations, history, search)
│   ├── rag.py                  # PDF loading, chunking, FAISS + BM25 hybrid search
│   ├── db.py                   # SQLite schema/build/query helpers
│   ├── sql_templates.py        # 15 pre-built SQL templates for common queries
│   ├── prompts/
│   │   └── system_instruction.txt  # Externalized system prompt with few-shot examples
│   └── fine_tuning/
│       ├── schema.py           # Vertex AI SFT dataset format helpers
│       ├── generate_dataset.py # Auto-generates Q&A pairs from real F1 database
│       └── tune.py             # Launches SFT job on Vertex AI
├── tests/                      # 83 unit tests (10 test modules)
├── deployment/
│   ├── deploy.py               # Vertex AI Agent Engine deploy script
│   └── terraform/              # GCP infrastructure as code
├── docs/                       # FIA PDFs + Kaggle CSV folder (input data)
├── vector_store/               # Generated FAISS artifacts (gitignored)
├── f1_data/                    # Generated SQLite artifact (gitignored)
├── f1_cache/                   # Semantic cache data (runtime, gitignored)
├── build_index.py              # Builds vector store + SQLite DB
├── .github/workflows/
│   ├── ci.yml                  # PR checks (ruff + tests)
│   └── deploy.yml              # Deploy pipeline on push to main
├── DEPLOY.md                   # Production deployment guide
├── DEVELOPMENT.md              # Development setup and architecture guide
├── Makefile
└── pyproject.toml
```

## Historical Database

The SQLite DB contains 14 tables with the following row counts:

| Table | Rows |
|---|---:|
| `circuits` | 77 |
| `constructors` | 212 |
| `drivers` | 861 |
| `races` | 1,125 |
| `results` | 26,759 |
| `qualifying` | 10,494 |
| `driver_standings` | 34,863 |
| `constructor_standings` | 13,391 |
| `constructor_results` | 12,625 |
| `lap_times` | 589,081 |
| `pit_stops` | 11,371 |
| `sprint_results` | 360 |
| `seasons` | 75 |
| `status` | 139 |

## Deployment and CI/CD

- **CI** (`.github/workflows/ci.yml`): Runs on PRs — ruff check, format check, unit tests
- **Deploy** (`.github/workflows/deploy.yml`): Runs on push to `main` — downloads artifacts, deploys to Agent Engine

For complete production setup (GCP, Terraform, secrets, manual deploy, smoke test):
- See [DEPLOY.md](./DEPLOY.md)

For development setup (local environment, architecture, testing, fine-tuning):
- See [DEVELOPMENT.md](./DEVELOPMENT.md)

## License

No license file is currently included in this repository.
