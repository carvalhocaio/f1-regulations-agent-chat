# F1 Regulations Agent Chat

AI assistant for Formula 1 that combines:
- Official **FIA 2026 regulations** (hybrid RAG over PDFs)
- Historical **F1 World Championship data (1950-2024)** in SQLite
- **Current and time-sensitive** information via Google Search

Built with Gemini + Vertex AI SDK and powered by local runtime wiring.

> This project is based on [f1-regulations-agent](https://github.com/carvalhocaio/f1-regulations-agent), the original CLI version.

## Core Capabilities

**Tools:**
- `search_regulations` — hybrid retrieval (FAISS semantic + BM25 keyword) over FIA 2026 Sections A-F
- `query_f1_history_template` — pre-built SQL templates for common F1 queries (champions, career stats, records, standings, head-to-head)
- `query_f1_history` — read-only SQL access to historical F1 data
- `google_search` — live web retrieval for current season/news
- `run_analytical_code` — restricted Code Execution sandbox for advanced analytics (feature-flagged)

**Intelligence layer:**
- **Model routing** — simple queries go to Flash (or fine-tuned Flash), complex ones stay on Pro
- **Semantic cache** — near-instant responses for repeated/similar questions (cosine similarity > 0.92)
- **Session corrections** — detects when users correct the agent (PT/EN) and avoids repeating mistakes
- **Local sessions (A2)** — in-memory session identity (`user_id` + `session_id`) for runtime context
- **Memory Bank (A3)** — optional long-term memory retrieval/generation per user across sessions
- **Dynamic few-shot via Example Store (A5)** — retrieves similar examples of real errors at runtime (feature-flagged)
- **Code Execution sandbox (A6, restricted mode)** — allowlisted analytical templates for simulations/statistics (feature-flagged)
- **RAG Engine rollout (A4)** — `search_regulations` supports phased routing (`auto|local|vertex`) with automatic fallback to local hybrid RAG
- **Standardized resilience** — exponential backoff + jitter + circuit breaker for transient 429/503 failures in runtime/tools
- **Strict tool contract** — no compatibility alias, stricter argument validation, and structured tool errors for invalid calls
- **Tool validation telemetry** — per-tool/per-error counters in runtime logs (e.g. `tool_validation_error`)
- **Runtime temporal context** — injects current UTC date/year on every request to avoid stale year assumptions after deploy
- **Temporal reasoning** — automatically splits questions: `1950-2024` via SQLite, `2025+` via web search

## How It Works

```text
User question
    |
    v
[check_cache] -----> Cache HIT? Return cached answer (<200ms)
    |
[inject_runtime_temporal_context] --> Inject current UTC date/year (per request)
    |
[inject_corrections] --> Append session corrections to prompt
    |
[inject_long_term_memories] --> Inject relevant cross-session memories
    |
[inject_dynamic_examples] --> Retrieve similar corrected errors from Example Store
    |
[route_model] -----> Simple? -> Flash/Tuned | Complex? -> Pro
    |
    v
Agent Runtime (Gemini)
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
    |-- Current/live ----------> google_search(request)
    |                             -> Web results
    |
    |-- Advanced analytics ----> run_analytical_code(task_type, payload)
    |                             -> Agent Engine sandbox (restricted templates)
    |
    v
[detect_corrections] --> Store if user corrected the agent
    |
[sync_memory_bank] ---> Generate long-term memory from current session context
    |
[store_cache] --------> Cache the answer for future reuse
                         (time-sensitive/web queries are not reused via cache)
    |
    v
Unified answer + sources
```

## Session Contract (No Login)

For clients without authentication, keep sessions persistent by sending:

- `client_id` — stable browser/device id stored in localStorage or cookie
- `user_id` — optional; if absent, backend derives deterministic anonymous id (`anon-<hash(client_id)>`)
- `session_id` — optional on first call; required on follow-ups to resume context

Recommended flow:

1. First call: send `client_id`, omit `session_id`
2. Backend creates Vertex session with TTL and returns `session_id`
3. Client stores `session_id` and sends it on subsequent calls

This enables cross-request persistence for callback state (for example, correction memory).

## WebSocket Streaming Contract (P7)

Bridge endpoint: `ws://<host>:8001/ws/chat`

Client -> server messages:
- `{"type":"input","input":"...","request_id":"...","user_id":"...","session_id":"..."}`
- `{"type":"abort"}`
- `{"type":"ping"}`
- `{"type":"close"}`

Optional structured response control (critical routes only):
- include `"response_contract_id"` in `input` messages when machine-readable JSON is required
- supported contract IDs:
  - `sources_block_v1`
  - `comparison_table_v1`
- runtime validates JSON and schema after model generation; invalid payloads are replaced by contract-compatible fallback JSON and logged as `structured_response_validation`

Server -> client events (`stream_protocol_version=v1`):
- `turn_start`
- `delta`
- `turn_end`
- `error`

## Stack

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
cat > .env <<'EOF'
GOOGLE_API_KEY=your-key-here

# Optional (recommended default for local embedding support)
# GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2-preview

# Optional (production only). Keep unset locally unless you have
# access to a valid Vertex tuned endpoint.
# F1_TUNED_MODEL=projects/<PROJECT_NUMBER>/locations/us-central1/endpoints/<ENDPOINT_ID>

# Optional (A4 phased rollout). Keep local by default.
# F1_RAG_BACKEND=local           # local|auto|vertex|vector_search
# F1_RAG_CORPUS=projects/<PROJECT_NUMBER>/locations/us-central1/ragCorpora/<RAG_CORPUS_ID>
# F1_RAG_PROJECT_ID=<PROJECT_ID>
# F1_RAG_LOCATION=us-central1
# F1_RAG_TOP_K=5
# F1_RAG_VECTOR_DISTANCE_THRESHOLD=0.5
# F1_VECTOR_SEARCH_PARENT=projects/<PROJECT_ID>/locations/us-central1/collections/<COLLECTION_ID>
# F1_VECTOR_SEARCH_FIELD=embedding
# F1_VECTOR_SEARCH_TOP_K=5
# F1_VECTOR_SEARCH_OUTPUT_FIELDS=data_fields,metadata_fields

# Optional (A5 dynamic few-shot). Keep disabled by default.
# F1_EXAMPLE_STORE_ENABLED=false
# F1_EXAMPLE_STORE_NAME=projects/<PROJECT_NUMBER>/locations/us-central1/exampleStores/<EXAMPLE_STORE_ID>
# F1_EXAMPLE_STORE_TOP_K=3
# F1_EXAMPLE_STORE_MIN_SCORE=0.65

# Optional (A3 Memory Bank). Keep disabled by default.
# F1_MEMORY_BANK_ENABLED=false
# F1_MEMORY_BANK_PROJECT_ID=<PROJECT_ID>
# F1_MEMORY_BANK_LOCATION=us-central1
# F1_MEMORY_BANK_AGENT_ENGINE_NAME=projects/<PROJECT_NUMBER>/locations/us-central1/reasoningEngines/<AGENT_ENGINE_ID>
# F1_MEMORY_BANK_MAX_FACTS=5
# F1_MEMORY_BANK_FETCH_LIMIT=20
# F1_MEMORY_BANK_GENERATE_ON_CORRECTION_ONLY=true

# Optional (A6 restricted Code Execution). Keep disabled by default.
# F1_CODE_EXECUTION_ENABLED=false
# F1_CODE_EXECUTION_LOCATION=us-central1
# F1_CODE_EXECUTION_AGENT_ENGINE_NAME=projects/<PROJECT_NUMBER>/locations/us-central1/reasoningEngines/<AGENT_ENGINE_ID>
# F1_CODE_EXECUTION_SANDBOX_TTL_SECONDS=3600
# F1_CODE_EXECUTION_MAX_ROWS=500

# Optional (P6 throughput routing). Default is DSQ/pay-as-you-go.
# F1_VERTEX_LLM_REQUEST_TYPE=shared  # shared|dedicated

# Optional (Q3 structured outputs). Enabled by default.
# F1_STRUCTURED_RESPONSE_ENABLED=true

# Optional (P2 resilience defaults tuned for chat fail-fast).
# F1_LLM_RETRY_ENABLED=true
# F1_LLM_RETRY_ATTEMPTS=3
# F1_LLM_RETRY_INITIAL_DELAY_S=1.0
# F1_LLM_RETRY_MAX_DELAY_S=8.0
# F1_LLM_RETRY_EXP_BASE=2.0
# F1_LLM_RETRY_JITTER=0.35
# F1_RETRY_ENABLED=true
# F1_RETRY_MAX_ATTEMPTS=3
# F1_RETRY_INITIAL_DELAY_S=0.4
# F1_RETRY_MAX_DELAY_S=4.0
# F1_RETRY_EXP_BASE=2.0
# F1_RETRY_JITTER=0.35
# F1_CIRCUIT_ENABLED=true
# F1_CIRCUIT_FAILURE_THRESHOLD=5
# F1_CIRCUIT_OPEN_SECONDS=20
EOF

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
│   ├── agent.py                # Agent definition, tools, and callbacks
│   ├── callbacks.py            # Model routing, semantic cache, session corrections
│   ├── runner.py               # Runner wiring for local in-memory sessions
│   ├── sessions.py             # user_id/session_id normalization helpers
│   ├── streaming_protocol.py    # Stream event envelope (`stream_protocol_version=v1`)
│   ├── bidi.py                 # Helpers to convert bidi SDK events into protocol events
│   ├── websocket_bridge.py      # Framework-agnostic WebSocket <-> bidi bridge loop
│   ├── memory_bank.py          # Long-term memory retrieval/generation (A3)
│   ├── cache.py                # SemanticCache (FAISS + SQLite, TTL-based)
│   ├── code_execution.py       # Restricted analytical sandbox adapter (A6)
│   ├── tools.py                # Agent tools (regulations, history, search)
│   ├── rag.py                  # PDF loading, chunking, FAISS + BM25 hybrid search
│   ├── rag_vertex.py           # Vertex RAG adapter (A4 phased externalization)
│   ├── rag_vector_search.py    # Vertex Vector Search adapter (P8 candidate backend)
│   ├── db.py                   # SQLite schema/build/query helpers
│   ├── sql_templates.py        # 15 pre-built SQL templates for common queries
│   ├── prompts/
│   │   └── system_instruction.txt  # Externalized system prompt with few-shot examples
│   └── fine_tuning/
│       ├── schema.py           # Vertex AI SFT dataset format helpers
│       ├── generate_dataset.py # Auto-generates Q&A pairs from real F1 database
│       └── tune.py             # Launches SFT job on Vertex AI
├── tests/                      # Unit tests (routing, cache, tools, sessions, temporal logic)
├── deployment/
│   ├── deploy.py               # Vertex AI Agent Engine deploy script
│   ├── smoke_bidi_agent_engine.py # Bidi streaming smoke test (P7)
│   ├── benchmark_streaming_modes.py # Compare TTFT across query/streaming modes
│   ├── benchmark_retrieval_backends.py # Compare local/vertex/vector_search retrieval
│   ├── vector_search_bootstrap.py # Create Vector Search collection and ingest chunks
│   ├── websocket_bidi_server.py # FastAPI WebSocket bridge for interactive bidi chat
│   ├── rag_engine_ingest.py    # Vertex RAG corpus create/import helper
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
- **Deploy** (`.github/workflows/deploy.yml`): Runs on push to `main` with 3 stages — deploy candidate, run Gen AI Evaluation gate (dataset + rubric metrics), then promote to production only when gate passes

For complete production setup (GCP, Terraform, secrets, manual deploy, smoke test):
- See [DEPLOY.md](./DEPLOY.md)

For development setup (local environment, architecture, testing, fine-tuning):
- See [DEVELOPMENT.md](./DEVELOPMENT.md)

## License

No license file is currently included in this repository.
