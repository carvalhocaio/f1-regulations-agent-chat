import json
import logging
import os
import re

from f1_agent import db
from f1_agent.rag import hybrid_search
from f1_agent.sql_templates import TEMPLATES, resolve_template

logger = logging.getLogger(__name__)

_RAG_BACKEND_ENV = "F1_RAG_BACKEND"
_RAG_BACKEND_LOCAL = "local"
_RAG_BACKEND_VERTEX = "vertex"
_RAG_BACKEND_VECTOR_SEARCH = "vector_search"
_RAG_BACKEND_AUTO = "auto"

_FORBIDDEN_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)


def search_regulations(query: str) -> dict:
    """Search the FIA 2026 F1 Regulations for relevant information.

    Uses hybrid search (FAISS semantic + BM25 keyword) with reciprocal rank
    fusion for better results, especially when searching for specific article
    numbers or technical terms.

    Args:
        query: The search query about F1 regulations.
    """
    backend = _selected_rag_backend()

    if backend == _RAG_BACKEND_LOCAL:
        results = _search_regulations_local(query, k=5)
    elif backend == _RAG_BACKEND_VERTEX:
        results = _search_regulations_vertex(query, k=5)
        if not results:
            logger.warning(
                "Vertex RAG returned no results; falling back to local search"
            )
            results = _search_regulations_local(query, k=5)
    elif backend == _RAG_BACKEND_VECTOR_SEARCH:
        results = _search_regulations_vector_search(query, k=5)
        if not results:
            logger.warning(
                "Vector Search returned no results; falling back to local search"
            )
            results = _search_regulations_local(query, k=5)
    else:
        for candidate in (
            _search_regulations_vector_search,
            _search_regulations_vertex,
            _search_regulations_local,
        ):
            results = candidate(query, k=5)
            if results:
                break

    if not results:
        return {"status": "no_results", "message": "No relevant regulations found."}

    chunks = []
    for doc in results:
        chunk = {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "section": doc.metadata.get("section", "unknown"),
        }
        article = doc.metadata.get("article")
        if article:
            chunk["article"] = article
        chunks.append(chunk)

    return {"status": "success", "results": chunks}


def _selected_rag_backend() -> str:
    raw = os.environ.get(_RAG_BACKEND_ENV, _RAG_BACKEND_AUTO).strip().lower()
    if raw in {
        _RAG_BACKEND_LOCAL,
        _RAG_BACKEND_VERTEX,
        _RAG_BACKEND_VECTOR_SEARCH,
        _RAG_BACKEND_AUTO,
    }:
        return raw
    return _RAG_BACKEND_AUTO


def _search_regulations_local(query: str, k: int = 5):
    return hybrid_search(query, k=k)


def _search_regulations_vertex(query: str, k: int = 5):
    try:
        from f1_agent.rag_vertex import vertex_hybrid_search

        return vertex_hybrid_search(query, k=k)
    except Exception:
        logger.warning(
            "Vertex RAG unavailable; falling back to local search", exc_info=True
        )
        return []


def _search_regulations_vector_search(query: str, k: int = 5):
    try:
        from f1_agent.rag_vector_search import vector_search_retrieve

        return vector_search_retrieve(query, k=k)
    except Exception:
        logger.warning(
            "Vector Search unavailable; falling back to next backend", exc_info=True
        )
        return []


def search(query: str | None = None, request: str | None = None) -> dict:
    """Compatibility fallback for hallucinated `search` tool calls.

    This tool exists only to avoid runtime failures when the model tries to call
    `search` (generic name) instead of one of the explicit tools.
    """
    normalized_query = (query or request or "").strip()
    valid_tools = [
        "search_regulations",
        "query_f1_history",
        "google_search_agent",
        "run_analytical_code",
    ]

    if not normalized_query:
        return {
            "status": "invalid_tool_alias",
            "message": (
                "Tool `search` is not a valid data source. Retry with one of the"
                " valid tools: search_regulations(query),"
                " query_f1_history(sql_query), or"
                " google_search_agent(request), or"
                " run_analytical_code(task_type, payload)."
            ),
            "valid_tools": valid_tools,
        }

    return {
        "status": "invalid_tool_alias",
        "message": (
            "Tool `search` is a compatibility alias only. Retry immediately with"
            " exactly one valid tool: search_regulations(query),"
            " query_f1_history(sql_query), google_search_agent(request),"
            " or run_analytical_code(task_type, payload)."
        ),
        "valid_tools": valid_tools,
        "suggested_query": normalized_query,
    }


def query_f1_history_template(template_name: str, params: str = "{}") -> dict:
    """Execute a pre-built SQL template against the F1 historical database.

    Use this tool instead of query_f1_history when the question matches one of
    the available templates.  Templates produce correct, optimised SQL and
    avoid common mistakes (e.g. wrong JOINs for champions).

    AVAILABLE TEMPLATES:

    - driver_champions(year?, from_year?, to_year?)
      World Drivers' Champions.
    - constructor_champions(year?, from_year?, to_year?)
      World Constructors' Champions.
    - race_winners_by_country(country)
      All race winners for a country/circuit.
    - race_results_by_year_country(year, country)
      Full top-10 result for a GP in a given year + country.
    - driver_career_stats(driver_name)
      Career stats: wins, podiums, poles, championships.
    - driver_season_results(driver_name, year)
      All race results for a driver in a season.
    - head_to_head_teammates(driver1, driver2, year)
      Head-to-head between teammates in a season.
    - most_wins_all_time(limit=10)
      Top N drivers by wins.
    - most_poles_all_time(limit=10)
      Top N drivers by poles.
    - most_podiums_all_time(limit=10)
      Top N drivers by podiums.
    - most_constructor_wins(limit=10)
      Top N constructors by wins.
    - season_calendar(year)
      Race calendar for a season.
    - season_standings_final(year)
      Final driver standings for a season.
    - fastest_pit_stops_race(year, country, limit=10)
      Fastest pit stops in a race.
    - fastest_laps_race(year, country, limit=10)
      Fastest laps in a race.

    Args:
        template_name: Name of the template to use (see list above).
        params: JSON string with template parameters,
                e.g. '{"year": 2023, "country": "Brazil"}'.
    """
    try:
        parsed_params = json.loads(params) if params else {}
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON in params: {e}"}

    try:
        sql = resolve_template(template_name, **parsed_params)
    except (KeyError, ValueError) as e:
        return {
            "status": "error",
            "message": str(e),
            "available_templates": list(TEMPLATES.keys()),
        }

    try:
        results = db.execute_query(sql)
    except Exception as e:
        return {"status": "error", "message": f"SQL error: {e}"}

    if not results:
        return {
            "status": "no_results",
            "message": "Query returned no results.",
            "results": [],
            "row_count": 0,
        }

    truncated = len(results) >= db.MAX_ROWS
    return {
        "status": "success",
        "results": results,
        "row_count": len(results),
        "truncated": truncated,
    }


def query_f1_history(sql_query: str) -> dict:
    """Query the Formula 1 World Championship historical database (1950-2024) using SQL.

    Write a SQLite SELECT query against the database. The database contains ~711K rows
    of official F1 championship data across 14 tables.

    DATABASE SCHEMA:
    ================

    LOOKUP TABLES:
    - circuits(circuitId PK, circuitRef, name, location, country, lat, lng, alt, url)
    - constructors(constructorId PK, constructorRef, name, nationality, url)
    - drivers(driverId PK, driverRef, number, code, forename, surname, dob, nationality, url)
    - seasons(year PK, url)
    - status(statusId PK, status)  -- e.g. "Finished", "Engine", "Collision", "+1 Lap"

    MAIN TABLES:
    - races(raceId PK, year, round, circuitId FK, name, date, time, url,
            fp1_date, fp1_time, fp2_date, fp2_time, fp3_date, fp3_time,
            quali_date, quali_time, sprint_date, sprint_time)
    - results(resultId PK, raceId FK, driverId FK, constructorId FK, number, grid,
              position, positionText, positionOrder, points, laps, time, milliseconds,
              fastestLap, rank, fastestLapTime, fastestLapSpeed, statusId FK)
    - qualifying(qualifyId PK, raceId FK, driverId FK, constructorId FK, number,
                 position, q1, q2, q3)
    - driver_standings(driverStandingsId PK, raceId FK, driverId FK, points,
                       position, positionText, wins)
    - constructor_standings(constructorStandingsId PK, raceId FK, constructorId FK,
                            points, position, positionText, wins)
    - constructor_results(constructorResultsId PK, raceId FK, constructorId FK, points, status)
    - lap_times(raceId PK/FK, driverId PK/FK, lap PK, position, time, milliseconds)
    - pit_stops(raceId PK/FK, driverId PK/FK, stop PK, lap, time, duration, milliseconds)
    - sprint_results(resultId PK, raceId FK, driverId FK, constructorId FK, number,
                     grid, position, positionText, positionOrder, points, laps, time,
                     milliseconds, fastestLap, fastestLapTime, statusId FK)

    RELATIONSHIPS:
    - races.circuitId -> circuits.circuitId
    - results.raceId -> races.raceId, results.driverId -> drivers.driverId
    - results.constructorId -> constructors.constructorId, results.statusId -> status.statusId
    - driver_standings.raceId -> races.raceId (standings after each race)
    - constructor_standings.raceId -> races.raceId
    - lap_times/pit_stops link to races and drivers

    QUERY TIPS:
    - Always JOIN with drivers/constructors/races tables to get readable names
    - For wins: WHERE position = 1 (in results table, position is INTEGER, use = 1 not = '1')
    - For podiums: WHERE position IN (1, 2, 3)
    - For DNFs: JOIN status and check status.statusId != 1 (1 = "Finished")
    - For season champion: use driver_standings joined with the LAST race of each year:
      SELECT ds.* FROM driver_standings ds
      JOIN (SELECT year, MAX(raceId) as lastRace FROM races GROUP BY year) lr
      ON ds.raceId = lr.lastRace WHERE ds.position = 1
    - Points, position, wins are numeric (not text)
    - NULL values exist where data is unavailable (e.g., old races without lap times)
    - Results are limited to 100 rows maximum

    IMPORTANT — RACE NAME CHANGES:
    - Race names have changed over the years! The same circuit/country can have different
      GP names across eras. Examples:
      * "São Paulo Grand Prix" (2021+) is the same event as "Brazilian Grand Prix" (1973-2019)
      * Races may be renamed due to sponsorship or rebranding
    - NEVER use exact match (= 'name') for race lookups. Instead:
      * Search by CIRCUIT: JOIN with circuits table and filter by circuits.country or
        circuits.location (e.g., WHERE c.country = 'Brazil')
      * Or use LIKE for partial matching (e.g., WHERE races.name LIKE '%Brazil%' OR
        races.name LIKE '%São Paulo%')
    - When asked about a GP's history, always search by circuit/country to capture ALL
      editions regardless of name changes
    - Similarly for drivers: use LIKE for name searches to handle variations

    Args:
        sql_query: A SQLite SELECT query to execute against the F1 historical database.
    """
    # Validate query is read-only
    stripped = sql_query.strip()
    if _FORBIDDEN_RE.match(stripped):
        return {
            "status": "error",
            "message": "Only SELECT queries are allowed. Write operations are not permitted.",
        }

    if not stripped.upper().startswith("SELECT"):
        return {
            "status": "error",
            "message": "Query must start with SELECT.",
        }

    try:
        results = db.execute_query(sql_query)
    except Exception as e:
        return {"status": "error", "message": f"SQL error: {e}"}

    if not results:
        return {
            "status": "no_results",
            "message": "Query returned no results.",
            "results": [],
            "row_count": 0,
        }

    truncated = len(results) >= db.MAX_ROWS

    return {
        "status": "success",
        "results": results,
        "row_count": len(results),
        "truncated": truncated,
    }
