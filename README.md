# F1 Technical Regulations Assistant
 
An AI assistant for the **FIA 2026 Formula 1 Technical Regulations**, built with:
 
- **[PydanticAI](https://ai.pydantic.dev/)** — type-safe LLM agents with structured outputs
- **[LangChain](https://python.langchain.com/)** — RAG pipeline (PDF loading + FAISS vector store)
- **[Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/)** — Google's LLM
- **[Rich](https://rich.readthedocs.io/)** — beautiful terminal interface

## How it works

```
User question
     │
     ▼
PydanticAI Agent (Gemini 2.5 Flash)
     │
     ├── @agent.tool: search_regulations(query)
     │        │
     │        ▼
     │   LangChain RAG
     │   (FAISS + gemini-embedding-2-preview)
     │        │
     │        ▼
     │   Relevant PDF chunks
     │
     ▼
RegulationAnswer (Pydantic model)
  ├── answer: str
  ├── references: list[RegulationReference]
  │     ├── article: str
  │     ├── title: str
  │     └── excerpt: str
  ├── confidence: float
  └── disclaimer: str | None
```

The key idea: instead of receiving a raw string from the LLM, you get a
**validated, structured Python object** — that's PydanticAI's superpower.
 
## Setup
 
### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install
 
```bash
git clone https://github.com/carvalhocaio/f1-regulations-agent
cd f1-regulations-agent
 
uv sync

source .venv/bin/activate.fish 
```

### 3. Configure API key
 
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```
 
Get your key at: https://aistudio.google.com/app/apikey

### 4. Download the PDF
 
Download the [FIA 2026 F1 Technical Regulations](https://www.fia.com/sites/default/files/fia_2026_formula_1_technical_regulations_issue_8_-_2024-06-24.pdf)
and save it as:
 
```
docs/fia_2026_f1_technical_regulations.pdf
```

### 5. Build the vector store
 
```bash
uv run build_index.py
```

This only needs to run once. The index is saved to `vector_store/`.
 
### 6. Run the assistant
 
```bash
uv run main.py
```

## Example questions
 
- *"What is the maximum power unit energy deployment per lap?"*
- *"What are the dimensions allowed for the front wing?"*
- *"Can the DRS be used during the formation lap?"*
- *"What materials are prohibited in car construction?"*
- *"What is the minimum weight of the car with the driver?"*

## Project structure
 
```
f1-regulations-agent/
├── docs/                  # Place the PDF here
├── vector_store/          # Auto-generated FAISS index
├── src/
│   ├── agent.py           # PydanticAI agent + tool definition
│   ├── models.py          # Pydantic output models
│   └── rag.py             # LangChain RAG pipeline
├── main.py                # Terminal chat interface
├── build_index.py         # One-time vector store builder
├── pyproject.toml
└── .env.example
```

## Related
 
- Tutorial: [Pydantic AI: Build Type-Safe LLM Agents in Python — Real Python](https://realpython.com/pydantic-ai/)
- Previous project: [pydantic-ai-type-safe-llm-agents-in-python](https://github.com/carvalhocaio/pydantic-ai-type-safe-llm-agents-in-python)
