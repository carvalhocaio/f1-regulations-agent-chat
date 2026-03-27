# F1 Regulations Agent Chat

AI assistant for Formula 1 that combines:
- Official FIA 2026 regulations (hybrid RAG over PDFs)
- Historical F1 World Championship data (1950-2024) in SQLite

This repository is based on
[f1-regulations-agent](https://github.com/carvalhocaio/f1-regulations-agent),
the original CLI project.

## Project Status (March 27, 2026)

- The runtime is actively maintained around `f1_agent/` and deployment scripts.
- The old local ADK web UI entrypoint was removed.
- `make run` does not start a server; it prints local validation guidance.
- Production target is Vertex AI Agent Engine.

## Core Runtime Capabilities

### Tools

- `search_regulations`
  - Hybrid search over FIA 2026 docs (FAISS semantic + BM25 keyword).
  - Backend mode via `F1_RAG_BACKEND`: `auto`, `local`, `vertex`, `vector_search`.
- `query_f1_history_template`
  - Uses 15 curated SQL templates for common historical queries.
- `query_f1_history`
  - Read-only raw SQL over local SQLite historical DB (SELECT-only; row cap 100).
- `run_analytical_code`
  - Restricted analytical sandbox with allowlisted task types:
    `summary_stats`, `what_if_points`, `distribution_bins`.
  - Feature-flagged (`F1_CODE_EXECUTION_ENABLED`).

### Intelligence Layer

- Model routing (`route_model`): simple routes to Flash/tuned Flash, complex stays on Pro.
- Semantic cache (`check_cache` / `store_cache`) with similarity-based retrieval.
- Runtime temporal context injection to prevent stale year assumptions.
- Relative-time resolution and local DB coverage enforcement (1950-2024).
- Session corrections memory (PT/EN pattern detection and reinjection).
- Dynamic few-shot injection from Example Store (optional).
- Structured response contracts and schema validation (optional/route-driven).
- Grounding policy callback (`observe` or `enforce` mode).
- Token preflight check with CountTokens API and progressive context truncation.
- Resilience layer: retries + exponential backoff/jitter + circuit breaker.

## Request Flow

```text
User question
    |
    v
before_model callbacks:
  1) check_cache
  2) inject_runtime_temporal_context
  3) inject_corrections
  4) inject_dynamic_examples
  5) route_model
  6) apply_throughput_request_type
  7) apply_grounding_policy
  8) apply_response_contract
  9) preflight_token_check
    |
    v
LLM + tools:
  - search_regulations
  - query_f1_history_template
  - query_f1_history
  - run_analytical_code (optional)
    |
    v
after_model callbacks:
  - log_context_cache_metrics
  - validate_structured_response
  - validate_grounding_outcome
  - detect_corrections
  - store_cache
```

## Session Contract (No Login Clients)

For persistent context across requests:

- Send stable `client_id` from browser/device.
- Optionally send `user_id`; otherwise backend derives `anon-<hash(client_id)>`.
- On first request, omit `session_id`.
- Store returned `session_id` and send it on subsequent calls.

Sessions are local in-memory (`InMemorySessionService`).

## WebSocket Streaming Contract

Bridge endpoint:

- `ws://<host>:8001/ws/chat`

Client to server message types:

- `{"type":"input","input":"...","request_id":"...","user_id":"...","session_id":"..."}`
- `{"type":"abort"}`
- `{"type":"ping"}`
- `{"type":"close"}`

Optional for structured JSON outputs on selected routes:

- Include `"response_contract_id"` in `input` message.
- Supported IDs:
  - `sources_block_v1`
  - `comparison_table_v1`

Server event envelope:

- `stream_protocol_version="v1"`
- Event types:
  - `turn_start`
  - `delta`
  - `tool_status`
  - `turn_end`
  - `error`

## Quick Start (Local)

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
# F1_RAG_BACKEND=auto
# F1_EXAMPLE_STORE_ENABLED=false
# F1_CODE_EXECUTION_ENABLED=false
# F1_STRUCTURED_RESPONSE_ENABLED=true
# F1_GROUNDING_POLICY_ENABLED=true
# F1_GROUNDING_POLICY_MODE=observe
# F1_VERTEX_LLM_REQUEST_TYPE=shared
# F1_PREFLIGHT_TOKEN_CHECK_ENABLED=false
EOF

# 4) Add source data to docs/
# - FIA 2026 regulation PDFs (Sections A-F)
# - Kaggle CSV folder "Formula 1 World Championship (1950 - 2024)"

# 5) Build artifacts
uv run build_index.py

# 6) Validate locally
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

`make run` is intentionally informational and prints the same guidance.

For full local setup details, see [DEVELOPMENT.md](./DEVELOPMENT.md).

## Configuration Reference (Most Used)

### Core

- `GOOGLE_API_KEY` / `GEMINI_API_KEY`: API key for Gemini.
- `GEMINI_EMBEDDING_MODEL`: embedding model used by RAG/cache.
- `F1_TUNED_MODEL`: optional tuned Flash endpoint for routing.

### Retrieval

- `F1_RAG_BACKEND`: `auto|local|vertex|vector_search`.
- `F1_RAG_CORPUS`, `F1_RAG_PROJECT_ID`, `F1_RAG_LOCATION`, `F1_RAG_TOP_K`.
- `F1_RAG_VECTOR_DISTANCE_THRESHOLD`.
- `F1_VECTOR_SEARCH_PARENT`, `F1_VECTOR_SEARCH_FIELD`,
  `F1_VECTOR_SEARCH_TOP_K`, `F1_VECTOR_SEARCH_OUTPUT_FIELDS`.

### Optional Feature Flags

- `F1_EXAMPLE_STORE_ENABLED`, `F1_EXAMPLE_STORE_NAME`,
  `F1_EXAMPLE_STORE_TOP_K`, `F1_EXAMPLE_STORE_MIN_SCORE`.
- `F1_CODE_EXECUTION_ENABLED`, `F1_CODE_EXECUTION_LOCATION`,
  `F1_CODE_EXECUTION_AGENT_ENGINE_NAME`,
  `F1_CODE_EXECUTION_SANDBOX_TTL_SECONDS`, `F1_CODE_EXECUTION_MAX_ROWS`.
- `F1_STRUCTURED_RESPONSE_ENABLED`.
- `F1_GROUNDING_POLICY_ENABLED`, `F1_GROUNDING_POLICY_MODE`,
  `F1_GROUNDING_TIME_SENSITIVE_SOURCE`.
- `F1_PREFLIGHT_TOKEN_CHECK_ENABLED`, `F1_PREFLIGHT_TOKEN_THRESHOLD`,
  `F1_PREFLIGHT_TOKEN_HARD_LIMIT`.
- `F1_VERTEX_LLM_REQUEST_TYPE` (`shared|dedicated`).

### Resilience and Cache

- LLM retry: `F1_LLM_RETRY_*`
- Tool retry/circuit breaker: `F1_RETRY_*`, `F1_CIRCUIT_*`
- Semantic cache tuning: `F1_SEMANTIC_CACHE_*`

## Data and Generated Artifacts

Gitignored runtime/generated directories:

- `vector_store/`: FAISS artifacts from FIA PDFs (`index.faiss`, `index.pkl`)
- `f1_data/`: SQLite database (`f1_history.db`)
- `f1_cache/`: semantic cache artifacts created at runtime

Current local DB state:

- Core Kaggle schema tables: 14
- Additional ingestion/support tables: `api_*` tables are present
  (`api_entity_map`, `api_drivers_profile`, `api_race_fastestlaps`, etc.)

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

## Deployment and CI/CD (Current Workflows)

- CI workflow: `.github/workflows/ci.yml`
  - Trigger: pull requests to `main`
  - Steps: `uv sync --frozen --extra dev`, `ruff check`,
    `ruff format --check`, unit tests
- Deploy workflow: `.github/workflows/deploy.yml`
  - Trigger: push to `main`
  - Steps:
    1. Install deps
    2. Download `vector_store/` and `f1_data/` artifacts from GCS
    3. Generate `requirements-deploy.txt` with `uv export`
    4. Deploy via `deployment/deploy.py`
    5. Run smoke test via `deployment/smoke_agent_engine.py`
    6. Upload `deployment_metadata.json` as workflow artifact

For complete production setup, Terraform, secrets, and manual deploy paths,
see [DEPLOY.md](./DEPLOY.md).

## Useful Commands

```bash
# Unit tests
uv run python -m unittest discover -s tests -p "test_*.py" -v

# Deploy manually (example)
uv run python deployment/deploy.py --help

# Smoke test deployed agent
uv run python deployment/smoke_agent_engine.py --help

# Optional bidi smoke test
uv run python deployment/smoke_bidi_agent_engine.py --help

# Optional local websocket bridge to a deployed agent
uv run python deployment/websocket_bidi_server.py --help
```

## Limitations and Notes

- Historical local DB coverage is 1950-2024.
- Questions requiring 2025+ or live standings/results are outside local DB.
- `query_f1_history` blocks write statements and limits output row count.
- Code execution is restricted to allowlisted analytical tasks only.

## License

No license file is currently included in this repository.
