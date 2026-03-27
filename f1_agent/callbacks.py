"""
ADK callbacks for model routing, semantic caching, and session corrections.

- ``route_model``: before-model callback that routes simple queries to Flash.
- ``check_cache`` / ``store_cache``: semantic answer cache callbacks.
- ``inject_corrections``: before-model callback that injects stored user
  corrections into the system instruction so the model doesn't repeat mistakes.
- ``detect_corrections``: after-model callback (actually runs on user turn)
  that detects correction patterns and stores them in session state.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timezone

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from f1_agent.cache import SemanticCache
from f1_agent.example_store import build_dynamic_examples_addendum

logger = logging.getLogger(__name__)

_DB_MAX_YEAR = 2024

# ── Model routing ───────────────────────────────────────────────────────

# Fine-tuned Flash model for F1 queries (Vertex AI endpoint).
# Set F1_TUNED_MODEL in .env to the tuned endpoint, or leave empty for base Flash.
FLASH_MODEL = os.environ.get("F1_TUNED_MODEL", "") or "gemini-2.5-flash"

# Patterns that indicate a *complex* question (keep on Pro)
_COMPLEX_PATTERNS: list[re.Pattern] = [
    # Multi-tool / comparative / analytical phrasing
    re.compile(r"(?:^|\s)(compar|vs\.?|versus|diferenç|difference)", re.I),
    re.compile(r"\b(analy[sz]|anális|evolv|evolu[çc]|evolution|trend)", re.I),
    re.compile(r"\b(before and|antes e|como mudou|how.+chang)\b", re.I),
    # Temporal reasoning across DB + web boundary
    re.compile(r"\b(últim[oa]s|last|recent)\s+\d+\b", re.I),
    # Regulation + historical mix
    re.compile(
        r"\b(regulament|regulation)\b.+\b(histor|season|champion|driver)\b", re.I
    ),
    re.compile(
        r"\b(histor|season|champion|driver)\b.+\b(regulament|regulation)\b", re.I
    ),
    # Open-ended / opinion / explanation
    re.compile(r"\b(por que|why|explain|expliqu|how does|como funciona)\b", re.I),
]


def _is_complex_question(text: str) -> bool:
    """Return True if the question likely requires Pro-level reasoning."""
    return any(p.search(text) for p in _COMPLEX_PATTERNS)


def _extract_user_text(callback_context, llm_request) -> str:
    """Extract the latest user message text from the request."""
    # Prefer callback_context.user_content (the original user message)
    user_content = getattr(callback_context, "user_content", None)
    if user_content and user_content.parts:
        texts = [p.text for p in user_content.parts if p.text]
        if texts:
            return " ".join(texts)

    # Fallback: scan llm_request.contents for the last user message
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            texts = [p.text for p in content.parts if p.text]
            if texts:
                return " ".join(texts)

    return ""


def _current_year() -> int:
    """Return current UTC year for temporal reasoning decisions."""
    return datetime.now(timezone.utc).year  # noqa: UP017 (py310 runtime)


def _current_date() -> date:
    """Return current UTC date for season-phase decisions."""
    return datetime.now(timezone.utc).date()  # noqa: UP017 (py310 runtime)


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
        " use google_search_agent.\n"
        f"- For data from {current_year} (ongoing season):"
        " use google_search_agent.\n"
        "- CRITICAL: Your training data may be outdated. If your training"
        f" data says '{last_completed} season hasn't concluded' or similar,"
        f" THAT IS WRONG. Trust THIS instruction: the {last_completed}"
        " season is over. When google_search_agent returns results about"
        " post-2024 seasons, TRUST and REPORT those results."
        " Do NOT contradict them with your training knowledge.\n"
        "- Do not describe completed seasons as future or ongoing events."
    )


# ── Temporal pre-resolution ────────────────────────────────────────────

# Patterns to detect relative temporal expressions and resolve to years
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


def _resolve_temporal_references(user_text: str) -> str | None:
    """Resolve relative temporal expressions to concrete years.

    Returns a prompt addendum with the resolution, or None if no
    relative temporal expressions were detected.
    """
    if not user_text:
        return None

    current_year = _current_year()
    current_date = _current_date()
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
                f"- Year {last_completed} is NOT in the database"
                " → use google_search_agent"
            )
        else:
            tool_hints.append(
                f"- Year {last_completed} is in the database"
                " → use query_f1_history_template or query_f1_history"
            )

    # "últimas N temporadas" / "last N seasons"
    m = _LAST_N_SEASONS_RE.search(user_text)
    if m:
        n = int(m.group(2))
        from_year = current_year - n
        to_year = last_completed
        resolutions.append(
            f"- 'Last {n}' = seasons {from_year} through {to_year} (all completed)"
        )
        db_years = [y for y in range(from_year, to_year + 1) if y <= _DB_MAX_YEAR]
        web_years = [y for y in range(from_year, to_year + 1) if y > _DB_MAX_YEAR]
        if db_years and web_years:
            tool_hints.append(
                f"- DB (query_f1_history_template): {db_years[0]}-{db_years[-1]}"
            )
            tool_hints.append(
                f"- Web (google_search_agent): {web_years[0]}-{web_years[-1]}"
            )
            tool_hints.append("- Combine BOTH into a single unified answer")
        elif web_years:
            tool_hints.append(
                f"- All years ({web_years[0]}-{web_years[-1]}) need google_search_agent"
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
            tool_hints.append(f"- {last_completed} champion: use google_search_agent")

    # "esta temporada" / "this season"
    if _THIS_SEASON_RE.search(user_text):
        resolutions.append(f"- 'This/current season' = {current_year} (may be ongoing)")
        tool_hints.append(f"- {current_year} season: use google_search_agent")

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
            "- Use google_search_agent and return the event DATE and YEAR"
            " to prove recency"
        )
        tool_hints.append(
            "- Never claim a season/event before current year is still in the future"
        )

    if _NEXT_EVENT_RESOLUTION_RE.search(user_text) and not _YEAR_RE.search(user_text):
        resolutions.append(
            f"- Missing year + 'next event' wording = next scheduled event in {current_year}"
        )
        tool_hints.append(
            "- Use google_search_agent for the current-year calendar and return"
            " event date"
        )

    if _CURRENT_STANDINGS_RE.search(user_text):
        resolutions.append(
            f"- Standings/leader request without explicit year defaults to {current_year}"
        )
        tool_hints.append(
            "- Use google_search_agent and verify if at least one race in the current"
            " season has already happened"
        )
        if current_date.month <= 2:
            tool_hints.append(
                f"- Pre-season guard ({current_date.isoformat()}): if no race has happened"
                f" yet in {current_year}, answer that the {current_year} season has not"
                " started and offer last season results"
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


def route_model(callback_context, llm_request):
    """Before-model callback: route simple queries to Flash."""
    user_text = _extract_user_text(callback_context, llm_request)
    if not user_text:
        return None

    if _is_complex_question(user_text):
        logger.debug("Routing to Pro (complex): %s", user_text[:80])
        return None  # keep the default model (Pro)

    logger.debug("Routing to Flash (simple): %s", user_text[:80])
    llm_request.model = FLASH_MODEL
    return None


# ── Semantic cache ──────────────────────────────────────────────────────

# Lazy singleton — created on first use
_cache: SemanticCache | None = None


def _get_cache() -> SemanticCache | None:
    """Return the shared SemanticCache, creating it on first call.

    Returns None if cache initialisation fails (e.g. missing API key in tests).
    """
    global _cache
    if _cache is None:
        try:
            _cache = SemanticCache()
        except Exception:
            logger.warning("Semantic cache unavailable — skipping.", exc_info=True)
            return None
    return _cache


def check_cache(callback_context, llm_request):
    """Before-model callback: return a cached answer if one exists."""
    cache = _get_cache()
    if cache is None:
        return None

    user_text = _extract_user_text(callback_context, llm_request)
    if not user_text:
        return None

    # Time-sensitive queries should prefer fresh tool calls.
    if _query_requires_web_data(user_text):
        return None

    hit = cache.get(user_text)
    if hit is None:
        return None

    logger.info("Cache HIT for: %s", user_text[:80])
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=hit)],
        )
    )


def store_cache(callback_context, llm_response):
    """After-model callback: store the answer in cache for future reuse."""
    cache = _get_cache()
    if cache is None:
        return None

    user_text = _extract_user_text(
        callback_context,
        # after_model_callback doesn't receive llm_request, so we use
        # callback_context.user_content only
        type("_Stub", (), {"contents": []})(),
    )
    if not user_text:
        return None

    # Extract model answer text
    if llm_response and llm_response.content and llm_response.content.parts:
        texts = [p.text for p in llm_response.content.parts if p.text]
        if texts:
            answer = "\n".join(texts)
            # Detect if answer used web data (short TTL)
            used_web = (
                "🌐" in answer
                or "google_search" in answer.lower()
                or _query_requires_web_data(user_text)
            )
            cache.put(user_text, answer, web_source=used_web)
            logger.debug("Cache STORE for: %s", user_text[:80])

    return None  # don't modify the response


# ── Session corrections ─────────────────────────────────────────────────

_CORRECTIONS_KEY = "f1_corrections"
_MAX_CORRECTIONS = 20

# Patterns that indicate the user is correcting the agent
_CORRECTION_PATTERNS: list[re.Pattern] = [
    # Portuguese
    re.compile(
        r"(na verdade|na real|está errado|tá errado|errou|não foi|"
        r"faltou|esqueceu|incorreto|correto é|certo é|"
        r"mas foi|mas na verdade|mas o certo)",
        re.I,
    ),
    # English
    re.compile(
        r"(actually|that'?s wrong|that'?s incorrect|you'?re wrong|"
        r"not correct|incorrect|you missed|you forgot|"
        r"the correct.+is|it was actually|no,?\s+it was|"
        r"wrong.+it'?s|that'?s not right)",
        re.I,
    ),
]


def _is_correction(text: str) -> bool:
    """Return True if the message looks like a user correction."""
    return any(p.search(text) for p in _CORRECTION_PATTERNS)


def _get_corrections(callback_context) -> list[str]:
    """Read the corrections list from session state."""
    state = getattr(callback_context, "state", None)
    if state is None:
        return []
    try:
        return list(state.get(_CORRECTIONS_KEY, []))
    except Exception:
        return []


def _store_correction(callback_context, correction: str) -> None:
    """Append a correction to session state, capping at MAX_CORRECTIONS."""
    state = getattr(callback_context, "state", None)
    if state is None:
        return

    corrections = _get_corrections(callback_context)
    corrections.append(correction)

    # Keep only the most recent corrections
    if len(corrections) > _MAX_CORRECTIONS:
        corrections = corrections[-_MAX_CORRECTIONS:]

    state[_CORRECTIONS_KEY] = corrections


def inject_corrections(callback_context, llm_request):
    """Before-model callback: inject stored corrections into the prompt.

    If the user has previously corrected the agent in this session,
    append those corrections to the system instruction so the model
    is aware of them and doesn't repeat the same mistakes.
    """
    corrections = _get_corrections(callback_context)
    if not corrections:
        return None

    corrections_text = "\n".join(f"- {c}" for c in corrections)
    addendum = (
        "\n\n## User corrections from this session\n"
        "The user has corrected you on the following points. "
        "Take these into account and do NOT repeat the same mistakes:\n"
        f"{corrections_text}"
    )

    llm_request.append_instructions([addendum])
    logger.debug("Injected %d corrections into prompt", len(corrections))
    return None


def inject_dynamic_examples(callback_context, llm_request):
    """Before-model callback: inject retrieved Example Store few-shots."""
    user_text = _extract_user_text(callback_context, llm_request)
    addendum, metadata = build_dynamic_examples_addendum(user_text)
    if not addendum:
        logger.debug(
            "Dynamic examples skipped (enabled=%s, configured=%s)",
            metadata.get("enabled"),
            metadata.get("store_configured"),
        )
        return None

    llm_request.append_instructions([addendum])
    logger.info(
        "Injected dynamic examples: count=%s top_similarity=%s",
        metadata.get("example_count"),
        metadata.get("top_similarity"),
    )
    return None


def inject_runtime_temporal_context(callback_context, llm_request):
    """Before-model callback: inject dynamic date/year guidance every request."""
    addenda = [_runtime_temporal_addendum()]

    user_text = _extract_user_text(callback_context, llm_request)
    resolution = _resolve_temporal_references(user_text)
    if resolution:
        addenda.append(resolution)

    llm_request.append_instructions(addenda)
    return None


def detect_corrections(callback_context, llm_response):
    """After-model callback: detect if the PREVIOUS user message was a correction.

    When the user corrects the agent, store the correction text in session
    state so it can be injected into future prompts via ``inject_corrections``.
    """
    user_text = _extract_user_text(
        callback_context,
        type("_Stub", (), {"contents": []})(),
    )
    if not user_text:
        return None

    if not _is_correction(user_text):
        return None

    # Store the user's correction verbatim (trimmed)
    correction = user_text.strip()
    if len(correction) > 500:
        correction = correction[:500] + "…"

    _store_correction(callback_context, correction)
    logger.info("Correction stored: %s", correction[:80])

    return None  # don't modify the response
