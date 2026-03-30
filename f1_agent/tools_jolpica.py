"""Jolpica/Ergast API tools for current season info and recent race results."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from datetime import UTC, date, datetime
from typing import Any

from f1_agent.tools_validation import _tool_error

logger = logging.getLogger(__name__)

# ── Season info tool ─────────────────────────────────────────────────────

_JOLPICA_URL = "https://api.jolpi.ca/ergast/f1/current.json"
_SEASON_CACHE_TTL_S = 3600  # 1 hour
_SEASON_CACHE_TIMEOUT_S = 3

_season_cache_lock = threading.Lock()
_season_cache: dict[str, Any] = {"data": None, "fetched_at": 0.0}


def _fetch_season_calendar() -> list[dict[str, Any]]:
    """Fetch current season calendar from Jolpica API with caching."""
    with _season_cache_lock:
        now = time.monotonic()
        if (
            _season_cache["data"] is not None
            and now - _season_cache["fetched_at"] < _SEASON_CACHE_TTL_S
        ):
            return _season_cache["data"]

    try:
        req = urllib.request.Request(
            _JOLPICA_URL,
            headers={"User-Agent": "f1-agent/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_SEASON_CACHE_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        races = payload["MRData"]["RaceTable"]["Races"]
    except Exception as exc:
        logger.warning("Jolpica API unavailable: %s", exc)
        with _season_cache_lock:
            if _season_cache["data"] is not None:
                return _season_cache["data"]
        return []

    with _season_cache_lock:
        _season_cache["data"] = races
        _season_cache["fetched_at"] = time.monotonic()
    return races


def get_current_season_info() -> dict[str, Any]:
    """Get the current F1 season calendar and identify which races have already happened.

    Returns the current season year, total number of races, and a breakdown
    of completed and upcoming races with their dates and circuit names.
    Use this tool to determine if the current F1 season has started and
    whether a specific Grand Prix has already taken place this year.
    """
    today = datetime.now(UTC).date()
    races = _fetch_season_calendar()

    if not races:
        return {
            "status": "unavailable",
            "message": (
                "Could not fetch season calendar. Use google_search as fallback."
            ),
        }

    season = races[0].get("season", str(today.year))
    completed = []
    upcoming = []

    for race in races:
        race_date_str = race.get("date", "")
        try:
            race_date = date.fromisoformat(race_date_str)
        except ValueError:
            continue

        entry = {
            "round": race.get("round"),
            "name": race.get("raceName"),
            "date": race_date_str,
            "circuit": race.get("Circuit", {}).get("circuitName"),
            "country": race.get("Circuit", {}).get("Location", {}).get("country"),
        }

        if race_date <= today:
            completed.append(entry)
        else:
            upcoming.append(entry)

    season_started = len(completed) > 0
    next_race = upcoming[0] if upcoming else None

    return {
        "status": "success",
        "season": season,
        "today": today.isoformat(),
        "season_started": season_started,
        "total_races": len(races),
        "completed_count": len(completed),
        "upcoming_count": len(upcoming),
        "completed_races": completed,
        "next_race": next_race,
        "upcoming_races": upcoming[:3],
    }


# ── Recent race results tool ────────────────────────────────────────────

_JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
_RESULTS_CACHE_TTL_S = 300  # 5 minutes (results may update shortly after race)
_results_cache_lock = threading.Lock()
_results_cache: dict[str, tuple[float, Any]] = {}


def _fetch_jolpica_json(path: str, cache_ttl: float = _RESULTS_CACHE_TTL_S) -> Any:
    """Fetch JSON from Jolpica API with per-path caching."""
    with _results_cache_lock:
        if path in _results_cache:
            fetched_at, data = _results_cache[path]
            if time.monotonic() - fetched_at < cache_ttl:
                return data

    url = f"{_JOLPICA_BASE}/{path}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "f1-agent/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Jolpica API request failed (%s): %s", url, exc)
        return None

    with _results_cache_lock:
        _results_cache[path] = (time.monotonic(), payload)
    return payload


def _parse_race_results(race: dict) -> dict[str, Any]:
    """Parse a single race result entry from Jolpica/Ergast format."""
    results = []
    for r in race.get("Results", []):
        driver = r.get("Driver", {})
        constructor = r.get("Constructor", {})
        entry = {
            "position": r.get("position"),
            "driver": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "constructor": constructor.get("name", ""),
            "status": r.get("status", ""),
            "points": r.get("points", "0"),
            "time": r.get("Time", {}).get("time", ""),
        }
        if r.get("FastestLap"):
            entry["fastest_lap"] = r["FastestLap"].get("Time", {}).get("time", "")
        results.append(entry)

    return {
        "race_name": race.get("raceName", ""),
        "round": race.get("round", ""),
        "date": race.get("date", ""),
        "circuit": race.get("Circuit", {}).get("circuitName", ""),
        "country": race.get("Circuit", {}).get("Location", {}).get("country", ""),
        "results": results,
    }


def search_recent_results(year: int, race_round: int = 0) -> dict[str, Any]:
    """Search for recent F1 race results from the Jolpica API (2025 onwards).

    This tool provides ACCURATE, STRUCTURED race results for seasons not
    covered by the historical database (2025+). Use it instead of google_search
    when you need race results, as it returns official classified results.

    IMPORTANT: Prefer this tool over google_search for race results from 2025+.
    Google search may return inaccurate or outdated snippets, especially for
    races that happened today. This tool returns the official classification.

    Args:
        year: The season year (2025 or later).
        race_round: The race round number. If 0 or omitted, returns the
                    latest race with results available.
    """
    if year < 2025:
        return _tool_error(
            tool_name="search_recent_results",
            code="INVALID_ARGUMENT",
            message=(
                f"Year {year} is covered by the historical database. "
                "Use query_f1_history_template or query_f1_history instead."
            ),
        )

    if race_round > 0:
        path = f"{year}/{race_round}/results.json"
    else:
        path = f"{year}/results.json?limit=100"

    payload = _fetch_jolpica_json(path)
    if payload is None:
        return {
            "status": "unavailable",
            "message": (
                "Could not fetch race results from Jolpica API. "
                "Use google_search as fallback."
            ),
        }

    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return {
            "status": "no_results",
            "message": f"No race results found for {year}"
            + (f" round {race_round}" if race_round else "")
            + ".",
        }

    if race_round > 0:
        parsed = _parse_race_results(races[0])
    else:
        # Return the latest race (last in the list)
        parsed = _parse_race_results(races[-1])

    return {
        "status": "success",
        "season": str(year),
        **parsed,
    }
