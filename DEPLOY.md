# Deploy — REST API (adk api_server)

Guide to run the F1 agent as a REST API using `adk api_server`.

## Architecture

```
adk api_server (FastAPI)
  ├── POST /run         ← synchronous agent execution
  ├── POST /run_sse     ← streaming via Server-Sent Events
  ├── WS   /run_live    ← live bidirectional streaming
  ├── GET  /health      ← health check
  ├── GET  /list-apps   ← list available agents
  └── Session CRUD      ← create, get, list, delete sessions

Agent runtime (f1_agent/)
  ├── Gemini 2.5 Pro (complex) + Flash (simple/tuned)
  ├── Hybrid RAG (FAISS + BM25) over FIA 2026 regulations
  ├── SQLite historical DB (1950-2024)
  ├── Semantic cache (FAISS + SQLite)
  └── Callback pipeline (routing, grounding, corrections, etc.)
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python package manager
- Python 3.10+
- A Gemini API key — get one at [Google AI Studio](https://aistudio.google.com/app/apikey)

## 1) Install dependencies

```bash
uv sync
```

## 2) Configure environment

Create a `.env` file in the project root:

```bash
# Required
GOOGLE_API_KEY=your-gemini-api-key

# Optional
# GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2-preview
# F1_RAG_BACKEND=auto
```

See [README.md](./README.md#configuration-reference-most-used) for the full configuration reference.

## 3) Generate data artifacts

The artifacts (`vector_store/` and `f1_data/`) are generated once locally.

```bash
# Ensure source data is in docs/
# - FIA 2026 regulation PDFs (Sections A-F)
# - "Formula 1 World Championship (1950 - 2024)" folder with Kaggle CSVs

# Generate indices
uv run build_index.py

# Verify artifacts were created
ls vector_store/   # index.faiss, index.pkl
ls f1_data/        # f1_history.db
```

## 4) Start the REST API

```bash
make api
```

Or directly:

```bash
uv run adk api_server \
  --host 127.0.0.1 \
  --port 8080 \
  --session_service_uri memory:// \
  --auto_create_session \
  .
```

The server starts at `http://localhost:8080`.

## 5) Test the API

### Health check

```bash
curl http://localhost:8080/health
```

### Synchronous query

```bash
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{
    "app_name": "f1_regulations_assistant",
    "user_id": "test-user",
    "new_message": {
      "role": "user",
      "parts": [{"text": "Who won the 2023 Formula 1 championship?"}]
    }
  }'
```

### Streaming (SSE)

```bash
curl -N -X POST http://localhost:8080/run_sse \
  -H "Content-Type: application/json" \
  -d '{
    "app_name": "f1_regulations_assistant",
    "user_id": "test-user",
    "new_message": {
      "role": "user",
      "parts": [{"text": "Explain DRS rules in the 2026 regulations"}]
    }
  }'
```

## ADK Web UI (alternative)

For development with a visual interface:

```bash
make dev
```

This starts the ADK web UI at `http://localhost:8000`.

## Notes

- **Sessions**: `--session_service_uri memory://` uses in-memory sessions (lost on restart). For persistence, ADK supports other backends.
- **Model routing**: Production uses `gemini-2.5-pro` for complex queries and Flash for simple queries. Routing is automatic via callbacks.
- **RAG backend**: `F1_RAG_BACKEND=auto` is the recommended default. It tries Vector Search, then Vertex RAG, then local FAISS+BM25 fallback.
- **Data artifacts**: If PDFs or CSVs are updated, re-run `build_index.py` locally.
