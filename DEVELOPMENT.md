# Development Guide

Complete guide for setting up the local development environment, understanding the architecture, running tests, and working with fine-tuning.

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

This generates:
- `vector_store/index.faiss` + `vector_store/index.pkl` — FAISS vector index from FIA regulation PDFs
- `f1_data/f1_history.db` — SQLite database from Kaggle CSVs (14 tables, 711K+ rows)

### 5. Run the assistant locally

```bash
make run
# or: uv run adk web f1_agent

# With managed Vertex sessions (persistent context)
# Requires GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_AGENT_ENGINE_ID
make run-managed
```

Opens the ADK web UI at `http://localhost:8000`.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | — | Gemini API key for LLM and embeddings |
| `GEMINI_EMBEDDING_MODEL` | No | `models/gemini-embedding-2-preview` | Embedding model for RAG and cache |
| `F1_TUNED_MODEL` | No | `gemini-2.5-flash` | Fine-tuned model endpoint (**production only** — see [Fine-Tuning](#fine-tuning-production-only)) |
| `F1_RAG_BACKEND` | No | `auto` | Regulations retrieval backend: `auto`, `local`, or `vertex` |
| `F1_RAG_CORPUS` | No | — | Vertex RAG corpus resource name (required when Vertex retrieval is used) |
| `F1_RAG_PROJECT_ID` | No | — | Explicit project for Vertex RAG client initialization |
| `F1_RAG_LOCATION` | No | — | Explicit location for Vertex RAG client initialization |
| `F1_RAG_TOP_K` | No | `5` | Top-k retrieved chunks for Vertex RAG |
| `F1_RAG_VECTOR_DISTANCE_THRESHOLD` | No | — | Optional retrieval distance threshold for Vertex RAG |
| `F1_EXAMPLE_STORE_ENABLED` | No | `false` | Enables dynamic few-shot retrieval from Example Store |
| `F1_EXAMPLE_STORE_NAME` | No | — | Example Store resource name: `projects/.../locations/.../exampleStores/...` |
| `F1_EXAMPLE_STORE_TOP_K` | No | `3` | Number of candidate examples retrieved per request |
| `F1_EXAMPLE_STORE_MIN_SCORE` | No | `0.65` | Similarity threshold for injecting an example |
| `F1_MEMORY_BANK_ENABLED` | No | `false` | Enables long-term memory retrieval/generation using Memory Bank |
| `F1_MEMORY_BANK_PROJECT_ID` | No | — | Explicit project id for Memory Bank client |
| `F1_MEMORY_BANK_LOCATION` | No | `us-central1` | Region for Memory Bank operations |
| `F1_MEMORY_BANK_AGENT_ENGINE_NAME` | No | — | Agent Engine resource name for memories API calls |
| `F1_MEMORY_BANK_MAX_FACTS` | No | `5` | Max long-term memories injected in each request |
| `F1_MEMORY_BANK_FETCH_LIMIT` | No | `20` | Max memory records fetched before filtering |
| `F1_MEMORY_BANK_GENERATE_ON_CORRECTION_ONLY` | No | `true` | Generate memory only when user explicitly corrects the agent |
| `F1_CODE_EXECUTION_ENABLED` | No | `false` | Enables restricted analytical Code Execution tool |
| `F1_CODE_EXECUTION_LOCATION` | No | `us-central1` | Region for Code Execution sandboxes (currently only `us-central1`) |
| `F1_CODE_EXECUTION_AGENT_ENGINE_NAME` | No | — | Agent Engine resource used as parent for sandbox operations |
| `F1_CODE_EXECUTION_SANDBOX_TTL_SECONDS` | No | `3600` | TTL for sandbox lifecycle in seconds |
| `F1_CODE_EXECUTION_MAX_ROWS` | No | `500` | Max list size accepted by analytical payload validators |
| `F1_VERTEX_LLM_REQUEST_TYPE` | No | `shared` | Vertex Gemini throughput route: `shared` (DSQ) or `dedicated` (Provisioned Throughput) |
| `F1_SEMANTIC_CACHE_SIMILARITY_THRESHOLD` | No | `0.92` | Minimum cosine similarity (via normalized inner product) for cache hit |
| `F1_SEMANTIC_CACHE_TOP_K` | No | `8` | ANN candidate count per cache lookup |
| `F1_SEMANTIC_CACHE_HNSW_M` | No | `32` | HNSW graph degree parameter |
| `F1_SEMANTIC_CACHE_HNSW_EF_SEARCH` | No | `64` | HNSW search breadth/recall parameter |
| `F1_SEMANTIC_CACHE_SWEEP_INTERVAL_S` | No | `600` | Periodic sweep interval (seconds) for expiry/pruning |
| `F1_SEMANTIC_CACHE_SWEEP_EVERY_OPS` | No | `500` | Operation-count trigger for expiry/pruning sweep |
| `F1_SEMANTIC_CACHE_MAX_ENTRIES` | No | `50000` | Max rows in semantic cache before low-priority pruning |

## Generated Artifacts

These directories are gitignored and must be generated locally:

| Directory | Source | Generated by | Purpose |
|-----------|--------|--------------|---------|
| `vector_store/` | FIA regulation PDFs | `build_index.py` | FAISS index for semantic search |
| `f1_data/` | Kaggle CSV dataset | `build_index.py` | SQLite database for historical queries |
| `f1_cache/` | — | Created at runtime | Semantic answer cache (FAISS + SQLite) |

## Architecture Overview

### Callback Pipeline

Every request passes through a pipeline of ADK callbacks:

```text
Before model:
  1. check_cache      — Return cached answer if similarity > 0.92
  2. inject_runtime_temporal_context — Inject current UTC date/year per request
  3. inject_corrections — Append user corrections from this session
  4. inject_long_term_memories — Inject relevant cross-session memories (A3)
  5. inject_dynamic_examples — Retrieve real-error few-shots from Example Store
  6. route_model      — Route to Flash/tuned (simple) or Pro (complex)
  7. apply_throughput_request_type — Set `X-Vertex-AI-LLM-Request-Type` (`shared|dedicated`)

After model:
  7. detect_corrections — Detect if the user corrected the agent (PT/EN)
  8. sync_memory_bank — Trigger long-term memory generation from session events
  9. store_cache      — Cache the answer (TTL: 30 days static, 24h web)

On error:
  10. handle_rate_limit — User-friendly fallback after retry exhaustion (429/503)

Runtime resilience layer:
  - LLM runtime retries via Gemini `HttpRetryOptions` (exponential backoff + jitter)
  - Tool/search retries + circuit breaker via `f1_agent.resilience`
```

### Model Routing

The `route_model` callback classifies questions using regex heuristics:

- **Simple** (~70% of queries) — factual lookups, single-tool questions → `gemini-2.5-flash` (or fine-tuned endpoint in production)
- **Complex** (~30%) — comparisons, temporal reasoning, multi-tool synthesis, open-ended questions → `gemini-2.5-pro`

Classification patterns for complex queries: comparisons (`vs`, `compare`, `diferença`), temporal ranges (`last N`, `últimos N`), regulation+history mix, open-ended (`why`, `how`, `por que`).

### Semantic Cache

`f1_agent/cache.py` provides near-instant responses for repeated questions:

- **Embedding model**: `GEMINI_EMBEDDING_MODEL` (default: `models/gemini-embedding-2-preview`)
- **Similarity threshold**: 0.92 (cosine)
- **Storage**: FAISS HNSW ANN (in-memory index) + SQLite (source of truth for answers/metadata)
- **TTL**: 30 days for historical/regulation data, 24 hours for web-sourced answers
- **Governance**: periodic sweep of expired rows + max entry cap with low-priority pruning
- **Freshness guard**: Questions that require live/post-2024 data bypass cache and force fresh tool calls
- **Location**: `f1_cache/` directory (created at runtime, gitignored)

### Hybrid RAG (Regulations)

`f1_agent/rag.py` combines two search strategies for regulation queries:

1. **FAISS semantic search** — embedding-based similarity
2. **BM25 keyword search** — term frequency-based matching
3. **Reciprocal Rank Fusion** (k=60) — merges and re-ranks results from both

Chunking is article-aware: separators prioritize `Article X.Y` boundaries, and article numbers are extracted into chunk metadata.

### External RAG rollout (A4)

`search_regulations` now supports a phased backend strategy:

- `F1_RAG_BACKEND=local` — always use local FAISS+BM25 (`f1_agent/rag.py`)
- `F1_RAG_BACKEND=vertex` — try Vertex RAG first; fallback to local if retrieval fails/returns empty
- `F1_RAG_BACKEND=auto` (default) — prefer Vertex when configured, with automatic local fallback

The Vertex adapter lives in `f1_agent/rag_vertex.py` and normalizes results to the same response shape used by existing tool consumers.

### Session Corrections

`f1_agent/callbacks.py` detects when users correct the agent:

- **Detection**: Regex patterns in Portuguese ("na verdade", "errou", "faltou") and English ("actually", "that's wrong", "you missed")
- **Storage**: Corrections stored in `callback_context.state["f1_corrections"]` (per-session)
- **Injection**: Before each LLM call, stored corrections are appended to the system instruction
- **Cap**: Maximum 20 corrections per session

### Dynamic Few-shot (A5)

`f1_agent/example_store.py` can retrieve semantically similar examples from a
Vertex AI Example Store and inject compact guidance before model execution:

- **Gate**: controlled by `F1_EXAMPLE_STORE_ENABLED`
- **Retrieval**: `search_examples(stored_contents_example_key=<user_text>)`
- **Selection**: top-k with `F1_EXAMPLE_STORE_TOP_K` + score filter via `F1_EXAMPLE_STORE_MIN_SCORE`
- **Safety**: failures are non-blocking; the request continues without dynamic examples

Manual curation workflow (v1):

1. Curate examples into `data/example_store/manual_examples.v1.jsonl`
2. Dry-run validation:

```bash
uv run python deployment/example_store_sync.py \
  --project-id <PROJECT_ID> \
  --location us-central1 \
  --dataset data/example_store/manual_examples.v1.jsonl \
  --dry-run
```

3. Sync to Example Store:

```bash
uv run python deployment/example_store_sync.py \
  --project-id <PROJECT_ID> \
  --location us-central1 \
  --dataset data/example_store/manual_examples.v1.jsonl \
  --example-store-name projects/<PROJECT_NUMBER>/locations/us-central1/exampleStores/<EXAMPLE_STORE_ID>
```

### Restricted Code Execution (A6)

`run_analytical_code` executes only allowlisted analytical templates in Agent
Engine Code Execution sandboxes:

- **Gate**: `F1_CODE_EXECUTION_ENABLED`
- **Tasks**: `summary_stats`, `what_if_points`, `distribution_bins`
- **Security model**: no arbitrary code input; payload is validated and mapped to predefined templates
- **Region**: `us-central1` only
- **Limits**: bounded payload size (`F1_CODE_EXECUTION_MAX_ROWS`) and sandbox TTL

Example local env snippet:

```bash
F1_CODE_EXECUTION_ENABLED=true
F1_CODE_EXECUTION_LOCATION=us-central1
F1_CODE_EXECUTION_AGENT_ENGINE_NAME=projects/<PROJECT_NUMBER>/locations/us-central1/reasoningEngines/<AGENT_ENGINE_ID>
F1_CODE_EXECUTION_SANDBOX_TTL_SECONDS=3600
F1_CODE_EXECUTION_MAX_ROWS=500
```

### Managed Sessions (A2)

`f1_agent/sessions.py` standardizes identity and session wiring for persistent context:

- `resolve_user_id(user_id, client_id)` — canonical user identity (supports anonymous mode)
- `build_session_identity(...)` — normalized `user_id/session_id` contract for clients
- `session_ttl_config(ttl_seconds)` — TTL payload helper for Vertex session creation
- `build_adk_session_service()` — uses `VertexAiSessionService` when env is configured; otherwise falls back to in-memory sessions

For no-login clients, keep a stable browser `client_id` and derive deterministic anonymous `user_id` (`anon-<hash>`).

### Long-term Memory Bank (A3)

`f1_agent/memory_bank.py` integrates Vertex Memory Bank with conservative defaults:

- **Gate**: `F1_MEMORY_BANK_ENABLED`
- **Inject**: before-model retrieval by `user_id` scope (cross-session)
- **Generate**: after-model trigger using current managed session events
- **Safety default**: `F1_MEMORY_BANK_GENERATE_ON_CORRECTION_ONLY=true`

This means v1 generates memory only when the user explicitly corrects the agent,
reducing noise and hallucination risk while still learning persistent preferences/facts.

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
uv run python -m unittest tests.test_fine_tuning -v
```

### Test Modules

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_sql_templates.py` | 12 | Template resolution, defaults, escaping, validation |
| `test_model_routing.py` | 14 | Complexity classification, user text extraction, routing |
| `test_semantic_cache.py` | 6 | Put/get, TTL, hit counts, clearing |
| `test_corrections.py` | 11 | Detection (PT/EN), storage, injection, cap |
| `test_hybrid_search.py` | 8 | Tokenization, article extraction |
| `test_fine_tuning.py` | 13 | Schema building, JSONL format, dataset generation |
| `test_agent_tool_contract.py` | 4 | Tool registration, prompt loading |
| `test_model_error_callback.py` | 4 | Rate limit handling (429/503) |
| `test_search_alias.py` | 3 | Compatibility alias fallback |
| `test_artifact_path_resolution.py` | 4 | Flat vs nested artifact layouts |
| `test_temporal_context.py` | 20+ | Dynamic year injection, temporal resolution and cache bypass |
| `test_sessions_contract.py` | 9 | Anonymous identity, TTL helpers, session service selection |
| `test_memory_bank.py` | 6+ | Long-term memory retrieval/generation and callback integration |
| `test_rag_backend.py` | 3 | A4 backend routing and local fallback behavior |

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

## Fine-Tuning (Production Only)

> **Fine-tuning is NOT required for local development.** The fine-tuned model is a Vertex AI endpoint used only in production. Locally, the model routing callback falls back to `gemini-2.5-flash` when `F1_TUNED_MODEL` is not set.

### How it works

The fine-tuning pipeline generates Q&A training pairs from the real F1 SQLite database, formats them as Vertex AI SFT JSONL, and launches a supervised fine-tuning job on `gemini-2.5-flash`.

The tuned model learns F1-specific patterns (tool selection, answer formatting, bilingual responses) and replaces the base Flash model in the routing callback for production deployments.

### Dataset format

Each training example is a text-only JSONL line with `{"contents": [user, model]}`:
- **User message**: Includes the system instruction as `[System: ...]` prefix
- **Model answer**: Includes `[Tool Use]` / `[Tool Results]` text representation of function calls, followed by the formatted answer

This format is required because `gemini-2.5-flash` SFT does not support `functionCall`/`functionResponse` modalities.

### Steps (for maintainers)

```bash
# 1. Generate the dataset (56 bilingual Q&A pairs from the real database)
uv run python -m f1_agent.fine_tuning.generate_dataset

# 2. Upload training and test splits to GCS
gsutil cp f1_agent/fine_tuning/dataset.train.jsonl gs://<BUCKET>/fine_tuning/dataset.train.jsonl
gsutil cp f1_agent/fine_tuning/dataset.test.jsonl gs://<BUCKET>/fine_tuning/dataset.test.jsonl

# 3. Launch the fine-tuning job
uv run python -m f1_agent.fine_tuning.tune \
    --project <PROJECT_ID> \
    --training-data gs://<BUCKET>/fine_tuning/dataset.train.jsonl \
    --validation-data gs://<BUCKET>/fine_tuning/dataset.test.jsonl

# 4. Once the job completes, store the endpoint in Secret Manager
echo -n "projects/<PROJECT_NUMBER>/locations/us-central1/endpoints/<ENDPOINT_ID>" | \
    gcloud secrets create f1-tuned-model --project <PROJECT_ID> --data-file=-

# 5. For local testing with the tuned model, add to .env:
# F1_TUNED_MODEL=projects/<PROJECT_NUMBER>/locations/us-central1/endpoints/<ENDPOINT_ID>
```

The deploy script (`deployment/deploy.py`) automatically reads the `f1-tuned-model` secret from Secret Manager and injects it as the `F1_TUNED_MODEL` environment variable in the Agent Engine runtime.

### Fine-tuning files

| File | Purpose |
|------|---------|
| `f1_agent/fine_tuning/schema.py` | TOOL_DECLARATIONS, `build_example()`, JSONL format helpers |
| `f1_agent/fine_tuning/generate_dataset.py` | 8 generators producing 56 Q&A pairs from real database |
| `f1_agent/fine_tuning/tune.py` | CLI to launch Vertex AI SFT job (`vertexai.tuning.sft.train`) |

## Troubleshooting

### `404 NOT_FOUND` for embeddings

If you see an error like:

`models/text-embedding-004 is not found for API version v1beta`

Use a supported embedding model in `.env`:

```bash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2-preview
```

The cache and RAG embedding paths both follow `GEMINI_EMBEDDING_MODEL`.

### `404 Not Found` when asking simple questions

If simple prompts fail and `.env` has `F1_TUNED_MODEL`, the local app may be trying
to call a production Vertex endpoint that is unavailable for your credentials/project.

For local development, keep `F1_TUNED_MODEL` unset so routing falls back to
`gemini-2.5-flash`.

## Key Modules Reference

| File | Purpose |
|------|---------|
| `f1_agent/agent.py` | ADK agent definition — model, tools, callbacks, instruction loading |
| `f1_agent/runner.py` | ADK runner setup with managed (Vertex) or in-memory sessions |
| `f1_agent/sessions.py` | Session identity normalization (`user_id`, `session_id`, `client_id`) and TTL helpers |
| `f1_agent/callbacks.py` | Before/after-model callbacks: routing, cache, corrections |
| `f1_agent/cache.py` | Semantic answer cache (FAISS + SQLite with TTL) |
| `f1_agent/tools.py` | Tool functions: `search_regulations`, `query_f1_history_template`, `query_f1_history` |
| `f1_agent/rag.py` | RAG pipeline: PDF loading, article-aware chunking, FAISS + BM25 hybrid search |
| `f1_agent/rag_vertex.py` | Vertex RAG adapter (`auto|local|vertex`) |
| `f1_agent/db.py` | SQLite DB: schema builder (from Kaggle CSVs), read-only query execution |
| `f1_agent/sql_templates.py` | 15 parameterized SQL templates for common F1 queries |
| `f1_agent/prompts/system_instruction.txt` | Externalized system prompt with few-shot examples |
| `deployment/deploy.py` | Vertex AI Agent Engine deploy/update script |
| `deployment/rag_engine_ingest.py` | Creates/imports Vertex RAG corpus from GCS PDFs |
| `build_index.py` | Generates `vector_store/` and `f1_data/` from source data in `docs/` |
