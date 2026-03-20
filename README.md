# F1 Regulations Agent Chat

An AI assistant for Formula 1 that combines **official FIA 2026 regulations** (via RAG) with **general F1 knowledge** (via Google Search), powered by [Google ADK](https://google.github.io/adk-docs/).

> Based on [f1-regulations-agent](https://github.com/carvalhocaio/f1-regulations-agent), the original CLI version built with PydanticAI + Rich.

## Stack

- **[Google ADK](https://google.github.io/adk-docs/)** — Agent Development Kit with built-in web UI
- **[Gemini 2.5 Pro](https://deepmind.google/technologies/gemini/)** — Google's LLM
- **[LangChain](https://python.langchain.com/)** — RAG pipeline (PDF loading + text splitting)
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search
- **[Google Search](https://google.github.io/adk-docs/)** — Real-time web search for general F1 knowledge

## How it works

```
User question
     |
     v
Google ADK Agent (Gemini 2.5 Pro)
     |
     |--- Regulation question? ---> search_regulations(query)
     |                                    |
     |                                    v
     |                              LangChain RAG
     |                              (FAISS + gemini-embedding-2-preview)
     |                                    |
     |                                    v
     |                              Relevant PDF chunks
     |                              (with section, page, source metadata)
     |
     |--- General F1 question? ---> google_search(query)
     |                                    |
     |                                    v
     |                              Web results
     |
     v
Answer with sources
```

The agent routes each question to the right tool: `search_regulations` for anything about the official FIA 2026 rules, and `google_search` for general F1 knowledge like history, standings, calendar, and current news.

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install

```bash
git clone https://github.com/carvalhocaio/f1-regulations-agent-chat
cd f1-regulations-agent-chat

uv sync
```

### 3. Configure API key

```bash
# Create a .env file and add your Google API key
echo "GOOGLE_API_KEY=your-key-here" > .env
```

Get your key at: https://aistudio.google.com/app/apikey

### 4. Download the PDFs

Download the six sections of the [FIA 2026 F1 Regulations](https://www.fia.com/regulation/type/110) and place them in the `docs/` directory:

- Section A — General Provisions
- Section B — Sporting
- Section C — Technical
- Section D — Financial (F1 Teams)
- Section E — Financial (Power Unit Manufacturers)
- Section F — Operational

### 5. Build the vector store

```bash
uv run build_index.py
```

This only needs to run once. The index is saved to `vector_store/`.

### 6. Run the assistant

```bash
adk web f1_agent
```

This starts the ADK web interface where you can chat with the agent.

## Example questions

**Regulations:**

- *"What is the maximum power unit energy deployment per lap?"*
- *"What are the dimensions allowed for the front wing?"*
- *"What materials are prohibited in car construction?"*
- *"What is the cost cap for F1 teams?"*

**General F1 knowledge:**

- *"What is the race calendar for this season?"*
- *"Who has the most world championships in F1 history?"*
- *"How does the sprint format work?"*
- *"What do the different flags mean in F1?"*

## Project structure

```
f1-regulations-agent-chat/
├── f1_agent/
│   ├── __init__.py        # Package exports
│   ├── agent.py           # Google ADK agent + tool routing
│   ├── tools.py           # search_regulations tool
│   └── rag.py             # LangChain RAG pipeline (FAISS)
├── docs/                  # FIA 2026 regulation PDFs (Sections A–F)
├── vector_store/          # Auto-generated FAISS index
├── build_index.py         # One-time vector store builder
└── pyproject.toml
```

## Regulation sections

The agent covers all six sections of the FIA 2026 F1 Regulations:

| Section | Description |
|---------|-------------|
| A | General Provisions |
| B | Sporting |
| C | Technical |
| D | Financial (F1 Teams) |
| E | Financial (Power Unit Manufacturers) |
| F | Operational |
