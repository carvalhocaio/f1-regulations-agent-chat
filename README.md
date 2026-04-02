# F1 Regulations Agent Chat

AI assistant for Formula 1 that combines:
- Official FIA 2026 regulations (hybrid RAG over PDFs)
- Historical F1 World Championship data (1950-2024) in SQLite

This repository is based on
[f1-regulations-agent](https://github.com/carvalhocaio/f1-regulations-agent),
the original CLI project.

## Project Status (April 2, 2026)

- The runtime is actively maintained around `f1_agent/`.
- Modular architecture: callbacks split into `cb_*` submodules, tools into `tools_*` submodules (facades preserve backward compatibility).
- Runs as a REST API via `adk api_server` (Google ADK).
- Endpoints: `POST /run`, `POST /run_sse` (streaming), `WebSocket /run_live`, `GET /health`.

## Core Runtime Capabilities

### Tools

- `search_regulations`
  - Hybrid search over FIA 2026 docs (FAISS semantic + BM25 keyword).
  - Backend mode via `F1_RAG_BACKEND`: `local`.
- `query_f1_history_template`
  - Uses 15 curated SQL templates for common historical queries.
- `query_f1_history`
  - Read-only raw SQL over local SQLite historical DB (SELECT-only; row cap 100).
- `get_current_season_info`
  - Current F1 season metadata.
- `search_recent_results`
  - Recent race results (web-sourced).
- `GoogleSearchTool` (optional)
  - Web search via Google Search. Enabled by default (`F1_GOOGLE_SEARCH_ENABLED=true`).

### Intelligence Layer

- Model routing (`route_model`): simple routes to Flash, complex stays on Pro. Configurable default model via `F1_DEFAULT_MODEL`.
- Semantic cache (`check_cache` / `store_cache`) with similarity-based retrieval.
- Runtime temporal context injection to prevent stale year assumptions.
- Relative-time resolution and local DB coverage enforcement (1950-2024).
- Session corrections memory (PT/EN pattern detection and reinjection).
- Structured response contracts and schema validation (optional/route-driven).
- Grounding policy callback (`observe` or `enforce` mode).
- Token preflight check with CountTokens API and progressive context truncation.
- Resilience layer: retries + exponential backoff/jitter + circuit breaker.
- Pro quota exhaustion detection with automatic Flash fallback.

## Request Flow

```text
User question
    |
    v
before_model callbacks:
  1) check_cache
  2) inject_runtime_temporal_context
  3) inject_corrections
  4) route_model
  5) apply_grounding_policy
  6) apply_response_contract
  7) preflight_token_check
    |
    v
LLM + tools:
  - search_regulations
  - query_f1_history_template
  - query_f1_history
  - get_current_season_info
  - search_recent_results
  - GoogleSearchTool (optional)
    |
    v
after_model callbacks:
  - log_context_cache_metrics
  - validate_structured_response
  - validate_grounding_outcome
  - detect_corrections
  - store_cache
    |
    v
on_model_error:
  - handle_rate_limit (429/503 → user-friendly fallback)
```

## Quick Start

```bash
# 1) Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Clone and install dependencies
git clone https://github.com/carvalhocaio/f1-regulations-agent-chat
cd f1-regulations-agent-chat
uv sync

# 3) Configure env
cat > .env <<'EOF'
# Required (either variable works; GEMINI_API_KEY is checked first)
GOOGLE_API_KEY=your-gemini-api-key
# GEMINI_API_KEY=your-gemini-api-key

# Optional local default
# GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2-preview

# Optional runtime toggles
# F1_RAG_BACKEND=local
# F1_STRUCTURED_RESPONSE_ENABLED=true
# F1_GROUNDING_POLICY_ENABLED=true
# F1_GROUNDING_POLICY_MODE=observe
# F1_PREFLIGHT_TOKEN_CHECK_ENABLED=false
EOF

# 4) Add source data to docs/
# - FIA 2026 regulation PDFs (Sections A-F)
# - Kaggle CSV folder "Formula 1 World Championship (1950 - 2024)"

# 5) Build artifacts
uv run build_index.py
# Exit codes: 0 = both artifacts built, 1 = partial/total failure

# 6) Run as REST API
make api
# Server starts at http://localhost:8080
```

For full local setup details, see [DEVELOPMENT.md](./DEVELOPMENT.md).

## Running the API

### Headless REST API (recommended)

```bash
make api
# or directly:
uv run adk api_server --host 127.0.0.1 --port 8080 --session_service_uri memory:// --auto_create_session .
```

### ADK Web UI (development)

```bash
make dev
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/run` | POST | Synchronous agent execution, returns list of events |
| `/run_sse` | POST | Streaming via Server-Sent Events |
| `/run_live` | WebSocket | Live bidirectional streaming |
| `/health` | GET | Health check |
| `/list-apps` | GET | List available agents |

Session CRUD endpoints are also available (create, get, list, delete).

## Configuration Reference (Most Used)

### Core

- `GOOGLE_API_KEY` / `GEMINI_API_KEY`: API key for Gemini.
- `GEMINI_EMBEDDING_MODEL`: embedding model used by RAG/cache.
- `F1_DEFAULT_MODEL`: default LLM model (default: `gemini-2.5-pro`).

### Retrieval

- `F1_RAG_BACKEND`: `local`.
- `F1_GOOGLE_SEARCH_ENABLED`: enable GoogleSearchTool (default: `true`).

### Optional Feature Flags

- `F1_STRUCTURED_RESPONSE_ENABLED`.
- `F1_GROUNDING_POLICY_ENABLED`, `F1_GROUNDING_POLICY_MODE`,
  `F1_GROUNDING_TIME_SENSITIVE_SOURCE`.
- `F1_PREFLIGHT_TOKEN_CHECK_ENABLED`, `F1_PREFLIGHT_TOKEN_THRESHOLD`,
  `F1_PREFLIGHT_TOKEN_HARD_LIMIT`.
- `F1_TOOL_METRICS_EXPORT_ENABLED`, `F1_TOOL_METRICS_PROJECT_ID`.

### Resilience and Cache

- LLM retry: `F1_LLM_RETRY_*` (`ENABLED`, `ATTEMPTS`, `INITIAL_DELAY_S`, `EXP_BASE`, `MAX_DELAY_S`, `JITTER`)
- Tool retry/circuit breaker: `F1_RETRY_*`, `F1_CIRCUIT_*`
- Semantic cache tuning: `F1_SEMANTIC_CACHE_*`

## Data and Generated Artifacts

Gitignored runtime/generated directories:

- `vector_store/`: FAISS artifacts from FIA PDFs (`index.faiss`, `index.pkl`)
- `f1_data/`: SQLite database (`f1_history.db`)
- `f1_cache/`: semantic cache artifacts created at runtime

Core table row counts in the current repository artifact:

| Table | Rows |
|---|---:|
| `circuits` | 114 |
| `constructors` | 232 |
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

## CI

- CI workflow: `.github/workflows/ci.yml`
  - Triggers: push to `main` + pull requests to `main`
  - **Lint job**: `ruff check` + `ruff format --check`
  - **Test job** (Python 3.11): `unittest discover`

## Useful Commands

```bash
# Unit tests
uv run python -m unittest discover -s tests -p "test_*.py" -v

# REST API server
make api

# ADK web UI
make dev
```

## Limitations and Notes

- Historical local DB coverage is 1950-2024.
- Questions requiring 2025+ or live standings/results are outside local DB.
- `query_f1_history` blocks write statements and limits output row count.
- Code execution is restricted to allowlisted analytical tasks only.

## License

No license file is currently included in this repository.
