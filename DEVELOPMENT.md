# Development Guide

Complete guide for setting up the local development environment, understanding the architecture, and running tests.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python package manager
- Python 3.10+
- A Gemini API key — get one at [Google AI Studio](https://aistudio.google.com/app/apikey)

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/carvalhocaio/f1-regulations-agent-chat
cd f1-regulations-agent-chat
uv sync
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```bash
# Required
GOOGLE_API_KEY=your-gemini-api-key

# Optional — override the default embedding model
# GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2-preview
```

> The app tries `GEMINI_API_KEY` first, then falls back to `GOOGLE_API_KEY`. Both work.

### 3. Add source data to `docs/`

**FIA 2026 Regulations** — download Sections A-F from:
- https://www.fia.com/regulation/type/110

**Historical F1 data** — download the Kaggle dataset:
- https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
- Place the extracted folder (starting with `Formula 1 ...`) inside `docs/`.

### 4. Build local artifacts

```bash
uv run build_index.py
```

Exit codes:
- `0` — both artifacts were built successfully (`vector_store/` and `f1_data/`)
- `1` — partial or total failure (check console output for which step failed)

This generates:
- `vector_store/index.faiss` + `vector_store/index.pkl` — FAISS vector index from FIA regulation PDFs
- `f1_data/f1_history.db` — SQLite database from Kaggle CSVs (14 tables, 711K+ rows)

### 5. Run the assistant locally

```bash
# Headless REST API (recommended)
make api

# ADK web UI (development)
make dev
```

`make api` starts a headless REST API at `http://localhost:8080` via `adk api_server`.
`make dev` starts the ADK web UI at `http://localhost:8000`.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| **Core** | | | |
| `GOOGLE_API_KEY` | Yes | — | Gemini API key for LLM and embeddings |
| `GEMINI_EMBEDDING_MODEL` | No | `models/gemini-embedding-2-preview` | Embedding model for RAG and cache |
| `F1_DEFAULT_MODEL` | No | `gemini-2.5-pro` | Default LLM model for the agent |
| **Retrieval** | | | |
| `F1_RAG_BACKEND` | No | `local` | Regulations retrieval backend (local-only) |
| `F1_GOOGLE_SEARCH_ENABLED` | No | `true` | Enable GoogleSearchTool for web queries |
| **Grounding** | | | |
| `F1_GROUNDING_POLICY_ENABLED` | No | `true` | Enables grounding policy callback for factual-critical routes |
| `F1_GROUNDING_POLICY_MODE` | No | `observe` | Grounding validation mode: `observe` (log only) or `enforce` (fallback on missing evidence) |
| `F1_GROUNDING_TIME_SENSITIVE_SOURCE` | No | `google` | Source used for time-sensitive grounding policy (Phase 1 currently supports `google`) |
| **Structured Responses** | | | |
| `F1_STRUCTURED_RESPONSE_ENABLED` | No | `false` | Enable structured response contracts and schema validation |
| **Token Preflight** | | | |
| `F1_PREFLIGHT_TOKEN_CHECK_ENABLED` | No | `false` | Enable token budget guard before LLM calls |
| `F1_PREFLIGHT_TOKEN_THRESHOLD` | No | — | Soft token threshold for progressive context truncation |
| `F1_PREFLIGHT_TOKEN_HARD_LIMIT` | No | — | Hard token limit; request is blocked if exceeded |
| **Observability** | | | |
| `F1_TOOL_METRICS_EXPORT_ENABLED` | No | `false` | Export tool validation errors as custom Cloud Monitoring metric |
| `F1_TOOL_METRICS_PROJECT_ID` | No | — | Project id used when exporting tool validation metrics (fallback: `GOOGLE_CLOUD_PROJECT`) |
| **Semantic Cache** | | | |
| `F1_SEMANTIC_CACHE_SIMILARITY_THRESHOLD` | No | `0.92` | Minimum cosine similarity (via normalized inner product) for cache hit |
| `F1_SEMANTIC_CACHE_TOP_K` | No | `8` | ANN candidate count per cache lookup |
| `F1_SEMANTIC_CACHE_HNSW_M` | No | `32` | HNSW graph degree parameter |
| `F1_SEMANTIC_CACHE_HNSW_EF_SEARCH` | No | `64` | HNSW search breadth/recall parameter |
| `F1_SEMANTIC_CACHE_SWEEP_INTERVAL_S` | No | `600` | Periodic sweep interval (seconds) for expiry/pruning |
| `F1_SEMANTIC_CACHE_SWEEP_EVERY_OPS` | No | `500` | Operation-count trigger for expiry/pruning sweep |
| `F1_SEMANTIC_CACHE_MAX_ENTRIES` | No | `50000` | Max rows in semantic cache before low-priority pruning |
| **LLM Resilience** | | | |
| `F1_LLM_RETRY_ENABLED` | No | `true` | Enable LLM retry with exponential backoff |
| `F1_LLM_RETRY_ATTEMPTS` | No | `3` | Max retry attempts for LLM calls |
| `F1_LLM_RETRY_INITIAL_DELAY_S` | No | `1.0` | Initial delay (seconds) before first retry |
| `F1_LLM_RETRY_EXP_BASE` | No | `2.0` | Exponential backoff base |
| `F1_LLM_RETRY_MAX_DELAY_S` | No | `8.0` | Maximum delay cap (seconds) |
| `F1_LLM_RETRY_JITTER` | No | `0.35` | Jitter factor for retry delays |

## Generated Artifacts

These directories are gitignored and must be generated locally:

| Directory | Source | Generated by | Purpose |
|-----------|--------|--------------|---------|
| `vector_store/` | FIA regulation PDFs | `build_index.py` | FAISS index for semantic search |
| `f1_data/` | Kaggle CSV dataset | `build_index.py` | SQLite database for historical queries |
| `f1_cache/` | — | Created at runtime | Semantic answer cache (FAISS + SQLite) |

## Architecture Overview

### Callback Pipeline

Every request passes through a callback pipeline. Each callback lives in its own `cb_*` submodule (the `callbacks.py` facade re-exports all of them for backward compatibility):

```text
Before model:
  1. check_cache      — Return cached answer if similarity > 0.92        (cb_semantic_cache)
  2. inject_runtime_temporal_context — Inject current UTC date/year       (cb_temporal)
  3. inject_corrections — Append user corrections from this session       (cb_corrections)
  4. route_model      — Route to Flash (simple) or Pro (complex)          (cb_model_routing)
  5. apply_grounding_policy — Attach grounding policy                     (cb_grounding)
  6. apply_response_contract — Attach response schema                     (cb_response_validation)
  7. preflight_token_check — Token budget guard                           (token_preflight)

After model:
  8.  log_context_cache_metrics — Context cache hit/miss logging          (callbacks — inline)
  9.  validate_structured_response — Schema validation                    (cb_response_validation)
  10. validate_grounding_outcome — Grounding evidence check               (cb_grounding)
  11. detect_corrections — Detect if the user corrected the agent         (cb_corrections)
  12. store_cache      — Cache the answer (TTL: 30d static, 24h live)     (cb_semantic_cache)

On error:
  13. handle_rate_limit — Pro quota exhaustion → Flash fallback; 429/503 → user-friendly message (agent.py)

Runtime resilience layer:
  - LLM runtime retries via Gemini `HttpRetryOptions` (exponential backoff + jitter)
  - Tool/search retries + circuit breaker via `f1_agent.resilience`
```

### Model Routing

The `route_model` callback classifies questions using regex heuristics:

- **Simple** (~70% of queries) — factual lookups, single-tool questions → `gemini-2.5-flash`
- **Complex** (~30%) — comparisons, temporal reasoning, multi-tool synthesis, open-ended questions → `gemini-2.5-pro`

Classification patterns for complex queries: comparisons (`vs`, `compare`, `diferença`), temporal ranges (`last N`, `últimos N`), regulation+history mix, open-ended (`why`, `how`, `por que`).

### Semantic Cache

`f1_agent/cache.py` provides near-instant responses for repeated questions:

- **Embedding model**: `GEMINI_EMBEDDING_MODEL` (default: `models/gemini-embedding-2-preview`)
- **Similarity threshold**: 0.92 (cosine)
- **Storage**: FAISS HNSW ANN (in-memory index) + SQLite (source of truth for answers/metadata)
- **TTL**: 30 days for historical/regulation data, 24 hours for time-sensitive/out-of-coverage prompts
- **Governance**: periodic sweep of expired rows + max entry cap with low-priority pruning
- **Freshness guard**: Questions that require live/post-2024 data bypass cache to avoid stale reuse
- **Location**: `f1_cache/` directory (created at runtime, gitignored)

### Hybrid RAG (Regulations)

`f1_agent/rag.py` combines two search strategies for regulation queries:

1. **FAISS semantic search** — embedding-based similarity
2. **BM25 keyword search** — term frequency-based matching
3. **Reciprocal Rank Fusion** (k=60) — merges and re-ranks results from both

Chunking is article-aware: separators prioritize `Article X.Y` boundaries, and article numbers are extracted into chunk metadata.

### Local RAG backend

`search_regulations` is local-only:

- `F1_RAG_BACKEND=local` — uses local FAISS+BM25 (`f1_agent/rag.py`)
- Any other configured value is ignored and falls back to local mode

### Session Corrections

`f1_agent/cb_corrections.py` detects when users correct the agent:

- **Detection**: Regex patterns in Portuguese ("na verdade", "errou", "faltou") and English ("actually", "that's wrong", "you missed")
- **Storage**: Corrections stored in `callback_context.state["f1_corrections"]` (per-session)
- **Injection**: Before each LLM call, stored corrections are appended to the system instruction
- **Cap**: Maximum 20 corrections per session

### Local Sessions (A2)

`f1_agent/sessions.py` standardizes identity and session wiring for persistent context:

- `resolve_user_id(user_id, client_id)` — canonical user identity (supports anonymous mode)
- `build_session_identity(...)` — normalized `user_id/session_id` contract for clients
- `session_ttl_config(ttl_seconds)` — shared TTL payload helper utility
- `build_adk_session_service()` — always uses local `InMemorySessionService`

For no-login clients, keep a stable browser `client_id` and derive deterministic anonymous `user_id` (`anon-<hash>`).

### SQL Templates

`f1_agent/sql_templates.py` provides 15 pre-built parameterized queries:

- `driver_champions`, `constructor_champions` — championship winners
- `driver_career_stats` — races, wins, podiums, poles, championships
- `race_results_by_year_country` — results by GP
- `most_wins_all_time`, `most_poles_all_time`, `most_podiums_all_time` — all-time records
- `season_standings_final` — end-of-season standings
- `head_to_head` — teammate comparison
- And more (pit stops, lap times, qualifying)

The LLM chooses between templates (`query_f1_history_template`) and free-form SQL (`query_f1_history`) based on the question.

## Running Tests

```bash
# Run all tests
uv run python -m unittest discover tests -v

# Run a specific test module
uv run python -m unittest tests.test_model_routing -v
```

### Test Modules

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_agent_tool_contract.py` | 4 | Tool registration, prompt loading |
| `test_artifact_path_resolution.py` | 4 | Flat vs nested artifact layouts |
| `test_corrections.py` | 11 | Detection (PT/EN), storage, injection, cap |
| `test_db.py` | 4 | Database connection, table existence |
| `test_grounding_policy.py` | 5 | Grounding policy enforcement |
| `test_hybrid_search.py` | 8 | Tokenization, article extraction, BM25 |
| `test_model_error_callback.py` | 4 | Rate limit handling (429/503), Pro quota fallback |
| `test_model_routing.py` | 14 | Complexity classification, user text extraction, routing |
| `test_rag_backend.py` | 2 | Local RAG backend selection behavior |
| `test_resilience.py` | 6 | Retry logic, circuit breaker state transitions |
| `test_response_contract.py` | 8 | Contract validation, schema enforcement |
| `test_season_info.py` | 4 | Current season info: completed/upcoming races, pre-season, API unavailable |
| `test_semantic_cache.py` | 6 | Put/get, TTL, hit counts, clearing |
| `test_sessions_contract.py` | 9 | Anonymous identity, TTL helpers, session service selection |
| `test_sql_templates.py` | 12 | Template resolution, defaults, escaping, validation |
| `test_temporal_context.py` | 20+ | Dynamic year injection, temporal resolution and cache bypass |
| `test_token_preflight.py` | 14 | Token counting, progressive context truncation |
| `test_tool_argument_validation.py` | 5 | Input validation, error handling for tool arguments |
| `test_tool_metrics.py` | 3 | Metrics export behavior, fail-open design |

## Linting

```bash
# Check for errors
uv run ruff check .

# Check formatting
uv run ruff format --check .

# Auto-fix
uv run ruff check --fix .
uv run ruff format .
```

## Troubleshooting

### `404 NOT_FOUND` for embeddings

If you see an error like:

`models/text-embedding-004 is not found for API version v1beta`

Use a supported embedding model in `.env`:

```bash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2-preview
```

The cache and RAG embedding paths both follow `GEMINI_EMBEDDING_MODEL`.

## Key Modules Reference

| File | Purpose |
|------|---------|
| `f1_agent/agent.py` | Agent definition — model, tools, callbacks, instruction loading, rate-limit error handler |
| `f1_agent/runner.py` | Runner setup with in-memory sessions |
| `f1_agent/sessions.py` | Session identity normalization (`user_id`, `session_id`, `client_id`) and TTL helpers |
| **Callbacks** | |
| `f1_agent/callbacks.py` | Facade — re-exports all callbacks from `cb_*` submodules |
| `f1_agent/cb_helpers.py` | Shared callback utilities (`_extract_user_text`, `_current_year`, `_current_date`, `_DB_MAX_YEAR`) |
| `f1_agent/cb_model_routing.py` | Model routing (`route_model`), Pro quota exhaustion tracking |
| `f1_agent/cb_temporal.py` | Temporal context injection, relative-time resolution, cache query classification |
| `f1_agent/cb_semantic_cache.py` | Semantic cache callbacks (`check_cache`, `store_cache`) |
| `f1_agent/cb_corrections.py` | Session corrections detection and injection |
| `f1_agent/cb_grounding.py` | Grounding policy (`observe`/`enforce` mode) |
| `f1_agent/cb_response_validation.py` | Structured response contracts and schema validation |
| **Tools** | |
| `f1_agent/tools.py` | Facade — re-exports all tools from `tools_*` submodules |
| `f1_agent/tools_validation.py` | Shared tool input validation and error helpers |
| `f1_agent/tools_rag.py` | `search_regulations` local retrieval wrapper |
| `f1_agent/tools_sql.py` | `query_f1_history`, `query_f1_history_template` |
| `f1_agent/tools_jolpica.py` | `get_current_season_info`, `search_recent_results` |
| **Infrastructure** | |
| `f1_agent/cache.py` | Semantic answer cache (FAISS + SQLite with TTL) |
| `f1_agent/env_utils.py` | Environment helpers (`env_bool`, `env_int`, `env_float`, `get_package_dir`) |
| `f1_agent/rag.py` | RAG pipeline: PDF loading, article-aware chunking, FAISS + BM25 hybrid search |
| `f1_agent/db.py` | SQLite DB: schema builder (from Kaggle CSVs), read-only query execution |
| `f1_agent/sql_templates.py` | 15 parameterized SQL templates for common F1 queries |
| `f1_agent/resilience.py` | Retry/backoff + circuit breaker utilities for LLM and tool calls |
| `f1_agent/response_contract.py` | Response contract definitions (`sources_block_v1`, `comparison_table_v1`) |
| `f1_agent/token_preflight.py` | Token preflight check: CountTokens API, progressive context truncation |
| `f1_agent/tool_metrics.py` | Cloud Monitoring export for tool validation errors (fail-open) |
| `f1_agent/prompts/system_instruction_static.txt` | Externalized system prompt with few-shot examples |
| `build_index.py` | Generates `vector_store/` and `f1_data/` from source data in `docs/` |
