# F1 Regulations Agent Chat

An AI assistant for Formula 1 that combines **official FIA 2026 regulations** (via RAG), **historical championship data 1950-2024** (via SQLite), and **real-time information** (via Google Search), powered by [Google ADK](https://google.github.io/adk-docs/).

> Based on [f1-regulations-agent](https://github.com/carvalhocaio/f1-regulations-agent), the original CLI version built with PydanticAI + Rich.

## Stack

- **[Google ADK](https://google.github.io/adk-docs/)** — Agent Development Kit with built-in web UI
- **[Gemini 2.5 Pro](https://deepmind.google/technologies/gemini/)** — Google's LLM
- **[LangChain](https://python.langchain.com/)** — RAG pipeline (PDF loading + text splitting)
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search
- **[SQLite](https://www.sqlite.org/)** — Historical F1 data from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) (~711K rows, 14 tables)
- **[Google Search](https://google.github.io/adk-docs/)** — Real-time web search for current F1 information

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
     |--- Historical/stats? -----> query_f1_history(sql_query)
     |                                    |
     |                                    v
     |                              SQLite database
     |                              (14 tables, 1950-2024 data)
     |                                    |
     |                                    v
     |                              Query results (read-only, max 100 rows)
     |
     |--- Current/live info? ----> google_search(query)
     |                                    |
     |                                    v
     |                              Web results
     |
     v
Answer with sources
```

The agent routes each question to the right tool:
- `search_regulations` for the official FIA 2026 rules
- `query_f1_history` for historical statistics and records (1950-2024)
- `google_search` for current season info, news, and general questions

The agent can also combine multiple tools in a single answer (e.g., comparing historical stats with 2026 regulation changes).

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

### 4. Download the data

**Regulations:** Download the six sections of the [FIA 2026 F1 Regulations](https://www.fia.com/regulation/type/110) and place them in the `docs/` directory:

- Section A — General Provisions
- Section B — Sporting
- Section C — Technical
- Section D — Financial (F1 Teams)
- Section E — Financial (Power Unit Manufacturers)
- Section F — Operational

**Historical data:** Download the [Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) dataset from Kaggle and place the folder inside `docs/`.

### 5. Build the indexes

```bash
uv run build_index.py
```

This builds both the FAISS vector store (`vector_store/`) and the SQLite database (`f1_data/f1_history.db`). Only needs to run once.

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

**Historical data:**

- *"Quantas vitórias Ayrton Senna teve?"*
- *"Quem mais venceu em Monza?"*
- *"Compare the win records of Hamilton and Schumacher"*
- *"Which constructor has the most championships?"*

**Current season:**

- *"What is the race calendar for this season?"*
- *"Who leads the championship?"*
- *"How does the sprint format work?"*

**Multiple tools:**

- *"Compare Schumacher's 2004 dominance with 2026 regulation changes"*
- *"How does Hamilton's record compare to the 2026 power unit rules?"*

## Project structure

```
f1-regulations-agent-chat/
├── f1_agent/
│   ├── __init__.py        # Package exports
│   ├── agent.py           # Google ADK agent + tool routing
│   ├── tools.py           # search_regulations + query_f1_history tools
│   ├── rag.py             # LangChain RAG pipeline (FAISS)
│   └── db.py              # SQLite database (schema, build, queries)
├── docs/                  # FIA 2026 regulation PDFs + Kaggle CSV dataset
├── vector_store/          # Auto-generated FAISS index
├── f1_data/               # Auto-generated SQLite database
├── build_index.py         # One-time index + database builder
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

## Historical database

The SQLite database contains 14 tables with complete F1 World Championship data (1950-2024):

| Table | Description | Rows |
|-------|-------------|------|
| `circuits` | Circuit details (name, location, country) | 77 |
| `constructors` | Constructor/team details | 212 |
| `drivers` | Driver details (name, nationality, DOB) | 861 |
| `races` | Race calendar (year, round, circuit, date) | 1,125 |
| `results` | Race results (position, points, time) | 26,759 |
| `qualifying` | Qualifying results (Q1, Q2, Q3 times) | 10,494 |
| `driver_standings` | Driver championship standings per race | 34,863 |
| `constructor_standings` | Constructor standings per race | 13,391 |
| `constructor_results` | Constructor points per race | 12,625 |
| `lap_times` | Individual lap times | 589,081 |
| `pit_stops` | Pit stop data (duration, lap) | 11,371 |
| `sprint_results` | Sprint race results | 360 |
| `seasons` | Season list with URLs | 75 |
| `status` | Race finish status codes | 139 |
