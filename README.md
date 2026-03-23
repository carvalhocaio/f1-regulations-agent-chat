# F1 Regulations Agent Chat

AI assistant for Formula 1 that combines:
- Official **FIA 2026 regulations** (RAG over PDFs)
- Historical **F1 World Championship data (1950-2024)** in SQLite
- **Current and time-sensitive** information via Google Search

Built with [Google ADK](https://google.github.io/adk-docs/) and powered by Gemini.

> This project is based on [f1-regulations-agent](https://github.com/carvalhocaio/f1-regulations-agent), the original CLI version.

## Core Capabilities

- `search_regulations`: semantic retrieval over FIA 2026 Sections A-F
- `query_f1_history`: read-only SQL access to historical F1 data
- `google_search`: live web retrieval for current season/news questions
- Multi-tool answers when a question spans historical + current + regulations data

The agent is explicitly instructed to split temporal questions across sources:
- `1950-2024` -> SQLite (`query_f1_history`)
- `2025+` or current season -> web (`google_search`)

## Stack

- [Google ADK](https://google.github.io/adk-docs/) - agent framework and local web UI
- [Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) - LLM (`gemini-2.5-flash`)
- [LangChain](https://python.langchain.com/) + [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF ingestion and chunking
- [FAISS](https://github.com/facebookresearch/faiss) - vector similarity search
- [SQLite](https://www.sqlite.org/) - historical F1 database
- Google ADK `GoogleSearchTool` - real-time web information
- Vertex AI Agent Engine (deployment target)
- Terraform + GitHub Actions (infrastructure and CI/CD)

## How It Works

```text
User question
    |
    v
ADK Agent (gemini-2.5-flash)
    |
    |-- Regulations question -----------> search_regulations(query)
    |                                      -> FAISS over FIA PDFs
    |
    |-- Historical/statistics ----------> query_f1_history(sql_query)
    |                                      -> SQLite (1950-2024, read-only SELECT)
    |
    |-- Current/live/time-sensitive ----> google_search(query)
    |                                      -> Web results
    |
    v
Unified answer + sources
```

## Local Setup

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/carvalhocaio/f1-regulations-agent-chat
cd f1-regulations-agent-chat

uv sync
```

### 3. Configure environment variables

Create `.env` in project root:

```bash
# Required (use either one)
GEMINI_API_KEY=your-key-here
# GOOGLE_API_KEY=your-key-here

# Optional
# GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2-preview
```

Notes:
- The app first tries `GEMINI_API_KEY`, then falls back to `GOOGLE_API_KEY`.
- Get API keys at https://aistudio.google.com/app/apikey

### 4. Add source data to `docs/`

Regulations: download FIA 2026 F1 Regulations Sections A-F from:
- https://www.fia.com/regulation/type/110

Historical data: download Kaggle dataset folder:
- https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
- Place the extracted folder (starting with `Formula 1 ...`) inside `docs/`.

### 5. Build local artifacts

```bash
uv run build_index.py
```

This generates:
- `vector_store/` (FAISS index)
- `f1_data/f1_history.db` (SQLite DB)

### 6. Run the assistant locally

```bash
uv run adk web f1_agent
```

## SQL Tool Constraints

`query_f1_history` is intentionally restricted:
- Only `SELECT` queries are allowed
- Write operations are blocked (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.)
- Results are capped at `100` rows

## Example Questions

Regulations:
- "What is the maximum power unit energy deployment per lap?"
- "What are the dimensions allowed for the front wing?"
- "What is the cost cap for F1 teams?"

Historical:
- "How many wins did Ayrton Senna have?"
- "Which constructor has the most championships?"
- "Compare Hamilton and Schumacher race wins."

Current/live:
- "What is the race calendar for this season?"
- "Who leads the championship right now?"

Cross-source:
- "Compare Schumacher's 2004 dominance with 2026 regulation changes."
- "Show the last 10 drivers' champions."

## Project Structure

```text
f1-regulations-agent-chat/
├── f1_agent/
│   ├── agent.py                # ADK agent definition and instructions
│   ├── tools.py                # search_regulations + query_f1_history
│   ├── rag.py                  # PDF loading, chunking, FAISS retrieval
│   └── db.py                   # SQLite schema/build/query helpers
├── docs/                       # FIA PDFs + Kaggle CSV folder (input data)
├── vector_store/               # Generated FAISS artifacts (gitignored)
├── f1_data/                    # Generated SQLite artifact package (gitignored)
├── build_index.py              # Builds vector store + SQLite DB
├── deployment/
│   ├── deploy.py               # Vertex AI Agent Engine deploy script
│   └── terraform/              # GCP infrastructure as code
├── .github/workflows/
│   ├── ci.yml                  # PR checks (ruff check + format check)
│   └── deploy.yml              # Deploy pipeline on push to main
├── DEPLOY.md                   # Full production deployment guide
└── pyproject.toml
```

## Historical Database (Current Local Dataset)

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

## Deployment and CI/CD (Summary)

- CI (`.github/workflows/ci.yml`):
  - Runs on PRs to `main`
  - Executes `uv run ruff check .` and `uv run ruff format --check .`
- Deploy (`.github/workflows/deploy.yml`):
  - Runs on push to `main` (environment: `production`)
  - Authenticates with GCP
  - Downloads `vector_store/` and `f1_data/` artifacts from Cloud Storage
  - Generates `requirements-deploy.txt`
  - Deploys/updates agent via `deployment/deploy.py`

For complete production setup (GCP, Terraform, secrets, manual deploy, smoke test):
- See [DEPLOY.md](./DEPLOY.md)

## License

No license file is currently included in this repository.
