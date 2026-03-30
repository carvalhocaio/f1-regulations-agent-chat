"""Temporal context injection and resolution for before-model callbacks."""

from __future__ import annotations

import logging
import re

from f1_agent.cb_helpers import (
    _DB_MAX_YEAR,
    _current_date,
    _current_year,
    _extract_user_text,
    _prepend_user_context,
)
from f1_agent.cb_model_routing import _is_complex_question

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_TIME_SENSITIVE_TERMS: list[re.Pattern] = [
    re.compile(r"\b(current|latest|live|today|now|this season|standings?)\b", re.I),
    re.compile(r"\b(atual|hoje|agora|temporada atual|classifica(?:ç|c)[aã]o)\b", re.I),
]

# Relative temporal expressions that likely need post-2024 data
_RELATIVE_TEMPORAL_RE: list[re.Pattern] = [
    re.compile(
        r"\b(últim[oa]s?|last|past|previous|passad[oa]|anterior)\s+"
        r"(temporada|season|campeonato|championship|ano|year)",
        re.I,
    ),
    re.compile(
        r"\b(atual|current|reigning|defend\w*)\s+"
        r"(campe[aã]o|champion|líder|leader)",
        re.I,
    ),
]

_EVENT_RECENCY_RE: list[re.Pattern] = [
    re.compile(
        r"\b(últim[oa]|mais recente|last|latest|most recent)\s+"
        r"(p[óo]dio|podium|resultado|result|vencedor|winner|gp|grande pr[eê]mio|"
        r"grand prix|corrida|race)",
        re.I,
    ),
    re.compile(
        r"\b(p[óo]dio|podium|resultado|result|vencedor|winner)\b.*"
        r"\b(últim[oa]|mais recente|last|latest|most recent)\b",
        re.I,
    ),
]

_NEXT_EVENT_RE: list[re.Pattern] = [
    re.compile(r"\b(pr[óo]xim[oa]|next|upcoming)\s+(gp|race|corrida|grand prix)", re.I),
]

_CURRENT_STANDINGS_RE = re.compile(
    r"\b(quem\s+[ée]\s+o\s+líder|lidera|leader|leading|standings?|"
    r"classifica(?:ç|c)[aã]o|campeonato\s+atual|championship\s+leader)\b",
    re.I,
)

_HISTORICAL_HINT_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b(histor|regulament|regulation|seasons?|championship)\b",
    re.I,
)

# ── Temporal pre-resolution patterns ─────────────────────────────────────

_LAST_SEASON_RE = re.compile(
    r"\b(últim[oa]|last|previous|passad[oa]|anterior)\s+"
    r"(temporada|season|campeonato|championship)",
    re.I,
)
_LAST_N_SEASONS_RE = re.compile(
    r"\b(últim[oa]s|last|past)\s+(\d+)\s+"
    r"(temporadas?|seasons?|anos?|years?|campeonatos?|championships?|"
    r"campe[oõ][ea]s?|champions?|vencedor[ea]s?|winners?)",
    re.I,
)
_CURRENT_CHAMPION_RE = re.compile(
    r"\b(atual|current|reigning|defend\w*)\s+"
    r"(campe[aã]o|champion)",
    re.I,
)
_THIS_SEASON_RE = re.compile(
    r"\b(esta?|this|current|atual)\s+(temporada|season|ano|year)\b",
    re.I,
)
_LAST_EVENT_RE = re.compile(
    r"\b(últim[oa]|mais recente|last|latest|most recent)\b.*"
    r"\b(gp|grande pr[eê]mio|grand prix|corrida|race|p[óo]dio|podium|"
    r"resultado|result|vencedor|winner)\b",
    re.I,
)
_NEXT_EVENT_RESOLUTION_RE = re.compile(
    r"\b(pr[óo]xim[oa]|next|upcoming)\b.*"
    r"\b(gp|grande pr[eê]mio|grand prix|corrida|race)\b",
    re.I,
)


def _query_requires_web_data(text: str) -> bool:
    """Return True if the question likely depends on post-2024/live data."""
    years = [int(y.group(0)) for y in _YEAR_RE.finditer(text)]
    if any(year > _DB_MAX_YEAR for year in years):
        return True

    if any(p.search(text) for p in _TIME_SENSITIVE_TERMS):
        return True

    if any(p.search(text) for p in _EVENT_RECENCY_RE):
        return True

    if any(p.search(text) for p in _NEXT_EVENT_RE):
        return True

    if _CURRENT_STANDINGS_RE.search(text):
        return True

    # Relative temporal expressions likely resolve to post-2024 seasons
    return any(p.search(text) for p in _RELATIVE_TEMPORAL_RE)


def _classify_cache_query(text: str) -> str:
    """Classify query type for semantic cache effectiveness metrics."""
    if _query_requires_web_data(text):
        return "time_sensitive"
    if _is_complex_question(text):
        return "complex"
    if _HISTORICAL_HINT_RE.search(text):
        return "historical"
    return "simple"


def _runtime_temporal_addendum() -> str:
    """Build authoritative temporal context to avoid stale year assumptions."""
    current_year = _current_year()
    last_completed = current_year - 1
    today_utc = _current_date().isoformat()
    return (
        "\n\n## Runtime temporal context — OVERRIDES YOUR TRAINING DATA\n"
        f"- Today (UTC): {today_utc}\n"
        f"- Current year: {current_year}\n"
        f"- The {last_completed} F1 season is FULLY COMPLETED"
        f" (ended Dec {last_completed}).\n"
        f"- All seasons from 1950 through {last_completed}"
        " are FINISHED, HISTORICAL seasons.\n"
        "- Historical DB coverage: 1950-2024 only.\n"
        f"- For data from 2025 through {last_completed}:"
        " use search_recent_results for race data or google_search for general info.\n"
        f"- For data from {current_year} (ongoing season):"
        " use google_search and get_current_season_info.\n"
        "- CRITICAL: Your training data may be outdated. If your training"
        f" data says '{last_completed} season hasn't concluded' or similar,"
        f" THAT IS WRONG. Trust THIS instruction: the {last_completed}"
        " season is over. Do NOT contradict this with stale assumptions.\n"
        f"- If unsure about {last_completed} results: USE google_search to"
        " verify. Do NOT say the season hasn't happened.\n"
        "- Do not describe completed seasons as future or ongoing events."
    )


def _resolve_temporal_references(user_text: str) -> str | None:
    """Resolve relative temporal expressions to concrete years.

    Returns a prompt addendum with the resolution, or None if no
    relative temporal expressions were detected.
    """
    if not user_text:
        return None

    current_year = _current_year()
    current_date = _current_date()  # noqa: F841
    last_completed = current_year - 1
    resolutions: list[str] = []
    tool_hints: list[str] = []

    # "última temporada" / "last season"
    if _LAST_SEASON_RE.search(user_text):
        resolutions.append(
            f"- 'Last/previous season' = **{last_completed}**"
            f" (COMPLETED — ended Dec {last_completed})"
        )
        if last_completed > _DB_MAX_YEAR:
            tool_hints.append(
                f"- Year {last_completed} is NOT in the local database;"
                " use search_recent_results for race data"
            )
        else:
            tool_hints.append(
                f"- Year {last_completed} is in the database"
                " → use query_f1_history_template or query_f1_history"
            )

    # "últimas N temporadas" / "last N seasons/winners/champions"
    m = _LAST_N_SEASONS_RE.search(user_text)
    if m:
        n = int(m.group(2))
        matched_term = m.group(3).lower()
        from_year = current_year - n
        to_year = last_completed
        # Event-specific terms (winners, podiums) may include current year
        _event_terms = (
            "vencedor",
            "winner",
            "podium",
            "pódio",
            "campe",
            "champion",
        )
        is_event_query = any(t in matched_term for t in _event_terms)
        resolutions.append(
            f"- 'Last {n}' = seasons {from_year} through {to_year} (all completed)"
        )
        if is_event_query:
            resolutions.append(
                f"- The {current_year} edition of this event MAY have already"
                " happened. Use get_current_season_info to check, and if so,"
                " include it in the answer."
            )
        db_years = [y for y in range(from_year, to_year + 1) if y <= _DB_MAX_YEAR]
        web_years = [y for y in range(from_year, to_year + 1) if y > _DB_MAX_YEAR]
        if db_years and web_years:
            tool_hints.append(
                f"- DB (query_f1_history_template): {db_years[0]}-{db_years[-1]}"
            )
            tool_hints.append(
                f"- Years {web_years[0]}-{web_years[-1]}: use search_recent_results"
                " to retrieve official race results"
            )
            tool_hints.append(
                "- REQUIRED: call search_recent_results for out-of-DB years,"
                " then combine with DB results into a single unified answer"
            )
        elif web_years:
            tool_hints.append(
                f"- All years ({web_years[0]}-{web_years[-1]}): use search_recent_results"
                " to retrieve official race results"
            )
        elif db_years:
            tool_hints.append(
                f"- All years ({db_years[0]}-{db_years[-1]}) are in the database"
            )

    # "campeão atual" / "current champion"
    if _CURRENT_CHAMPION_RE.search(user_text):
        resolutions.append(
            f"- 'Current/reigning champion' = the {last_completed} champion"
            f" (season COMPLETED)"
        )
        if last_completed > _DB_MAX_YEAR:
            tool_hints.append(
                f"- {last_completed} champion is outside DB coverage;"
                " use search_recent_results or google_search to find the answer"
            )

    # "esta temporada" / "this season"
    if _THIS_SEASON_RE.search(user_text):
        resolutions.append(f"- 'This/current season' = {current_year} (may be ongoing)")
        tool_hints.append(
            f"- {current_year} season is outside DB coverage;"
            " use google_search and get_current_season_info"
        )

    if _LAST_EVENT_RE.search(user_text) and not _YEAR_RE.search(user_text):
        resolutions.append(
            "- Missing year + 'last/latest' event wording"
            " = interpret as the LAST COMPLETED edition of that event"
        )
        resolutions.append(
            f"- In {current_year}, that usually means {last_completed} or {current_year}"
            " depending on event date"
        )
        tool_hints.append(
            "- Use get_current_season_info to check if the event happened in"
            f" {current_year}, then use google_search for the result"
        )
        tool_hints.append("- Return event DATE and YEAR to prove recency")
        tool_hints.append(
            "- Never claim a season/event before current year is still in the future"
        )

    if _NEXT_EVENT_RESOLUTION_RE.search(user_text) and not _YEAR_RE.search(user_text):
        resolutions.append(
            f"- Missing year + 'next event' wording = next scheduled event in {current_year}"
        )
        tool_hints.append(
            "- Use get_current_season_info to find the next scheduled race,"
            " then google_search for details"
        )

    if _CURRENT_STANDINGS_RE.search(user_text):
        resolutions.append(
            f"- Standings/leader request without explicit year defaults to {current_year}"
        )
        tool_hints.append(
            "- Use get_current_season_info to check if the season has started,"
            " then google_search for current standings"
        )

    if not resolutions:
        return None

    lines = [
        "\n\n## Temporal resolution (pre-computed — TRUST THIS)\n"
        "The user's question contains relative time references."
        " Resolved values:",
    ]
    lines.extend(resolutions)
    if tool_hints:
        lines.append("\nTool routing:")
        lines.extend(tool_hints)

    return "\n".join(lines)


def inject_runtime_temporal_context(callback_context, llm_request):
    """Before-model callback: inject dynamic date/year guidance every request.

    Inserts temporal context as user content to keep the system instruction
    stable for context caching.
    """
    parts = [_runtime_temporal_addendum()]

    user_text = _extract_user_text(callback_context, llm_request)
    resolution = _resolve_temporal_references(user_text)
    if resolution:
        parts.append(resolution)

    _prepend_user_context(llm_request, "\n\n".join(parts))
    return None
