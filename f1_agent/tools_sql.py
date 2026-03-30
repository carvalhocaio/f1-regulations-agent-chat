"""SQL query tools for the F1 historical database."""

from __future__ import annotations

import json
import logging
import re

from f1_agent import db
from f1_agent.sql_templates import TEMPLATES, resolve_template
from f1_agent.tools_validation import _normalize_non_empty_text, _tool_error

logger = logging.getLogger(__name__)

_FORBIDDEN_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)


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
    normalized_template_name = _normalize_non_empty_text(
        value=template_name,
        field_name="template_name",
        max_len=120,
    )
    if normalized_template_name is None:
        return _tool_error(
            tool_name="query_f1_history_template",
            code="INVALID_ARGUMENT",
            message="query_f1_history_template requires `template_name`.",
            details={"field": "template_name", "expected": "non-empty string"},
        )

    try:
        parsed_params = json.loads(params) if params else {}
    except json.JSONDecodeError as exc:
        return _tool_error(
            tool_name="query_f1_history_template",
            code="INVALID_ARGUMENT",
            message=f"Invalid JSON in `params`: {exc}",
            details={"field": "params", "expected": "JSON object string"},
        )

    if not isinstance(parsed_params, dict):
        return _tool_error(
            tool_name="query_f1_history_template",
            code="INVALID_ARGUMENT",
            message="`params` must be a JSON object.",
            details={"field": "params", "expected": "JSON object"},
        )

    template_spec = TEMPLATES.get(normalized_template_name)
    if template_spec is None:
        return _tool_error(
            tool_name="query_f1_history_template",
            code="INVALID_TEMPLATE",
            message=(
                f"Unknown template '{normalized_template_name}'."
                f" Valid templates: {', '.join(TEMPLATES.keys())}"
            ),
            details={"available_templates": sorted(TEMPLATES.keys())},
        )

    allowed_param_keys = set(template_spec.get("params", {}).keys())
    unknown_param_keys = sorted(
        k for k in parsed_params.keys() if k not in allowed_param_keys
    )
    if unknown_param_keys:
        return _tool_error(
            tool_name="query_f1_history_template",
            code="INVALID_ARGUMENT",
            message=(
                "Unknown parameter(s) for template "
                f"'{normalized_template_name}': {', '.join(unknown_param_keys)}"
            ),
            details={
                "field": "params",
                "unknown_keys": unknown_param_keys,
                "allowed_keys": sorted(allowed_param_keys),
            },
        )

    try:
        sql = resolve_template(normalized_template_name, **parsed_params)
    except (KeyError, TypeError, ValueError) as exc:
        return _tool_error(
            tool_name="query_f1_history_template",
            code="INVALID_ARGUMENT",
            message=str(exc),
            details={"available_templates": sorted(TEMPLATES.keys())},
        )

    try:
        results = db.execute_query(sql)
    except Exception as exc:
        return _tool_error(
            tool_name="query_f1_history_template",
            code="SQL_EXECUTION_ERROR",
            message=f"SQL error: {exc}",
        )

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
    stripped = _normalize_non_empty_text(
        value=sql_query,
        field_name="sql_query",
        max_len=6000,
    )
    if stripped is None:
        return _tool_error(
            tool_name="query_f1_history",
            code="INVALID_ARGUMENT",
            message="query_f1_history requires non-empty string `sql_query`.",
            details={"field": "sql_query", "expected": "non-empty string"},
        )

    # Validate query is read-only
    if _FORBIDDEN_RE.match(stripped):
        return _tool_error(
            tool_name="query_f1_history",
            code="READ_ONLY_ENFORCED",
            message="Only SELECT queries are allowed. Write operations are not permitted.",
        )

    if not stripped.upper().startswith("SELECT"):
        return _tool_error(
            tool_name="query_f1_history",
            code="INVALID_QUERY",
            message="Query must start with SELECT.",
        )

    if ";" in stripped.rstrip(";"):
        return _tool_error(
            tool_name="query_f1_history",
            code="INVALID_QUERY",
            message="Multiple SQL statements are not allowed.",
        )

    try:
        results = db.execute_query(stripped)
    except Exception as exc:
        return _tool_error(
            tool_name="query_f1_history",
            code="SQL_EXECUTION_ERROR",
            message=f"SQL error: {exc}",
        )

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
