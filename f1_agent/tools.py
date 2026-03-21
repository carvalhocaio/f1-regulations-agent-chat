import re

from f1_agent import db
from f1_agent.rag import get_vector_store

_FORBIDDEN_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)


def search_regulations(query: str) -> dict:
    """Search the FIA 2026 F1 Regulations for relevant information.

    Args:
        query: The search query about F1 regulations.
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=5)

    if not results:
        return {"status": "no_results", "message": "No relevant regulations found."}

    chunks = []
    for doc in results:
        chunks.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "section": doc.metadata.get("section", "unknown"),
            }
        )

    return {"status": "success", "results": chunks}


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
