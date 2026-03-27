"""
ADK callbacks for model routing, semantic caching, and session corrections.

- ``route_model``: before-model callback that routes simple queries to Flash.
- ``check_cache`` / ``store_cache``: semantic answer cache callbacks.
- ``inject_corrections``: before-model callback that injects stored user
  corrections into the system instruction so the model doesn't repeat mistakes.
- ``inject_long_term_memories``: before-model callback that injects cross-session
  Memory Bank facts for the same user.
- ``detect_corrections``: after-model callback (actually runs on user turn)
  that detects correction patterns and stores them in session state.
- ``sync_memory_bank``: after-model callback that triggers long-term memory
  generation from managed session history.
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
from f1_agent.memory_bank import (
    build_memory_addendum,
    generate_memories_from_session,
)
from f1_agent.memory_bank import (
    load_settings as load_memory_bank_settings,
)
from f1_agent.token_preflight import check_and_truncate as _check_and_truncate

logger = logging.getLogger(__name__)

_DB_MAX_YEAR = 2024
_DEFAULT_VERTEX_REQUEST_TYPE = "shared"
_VALID_VERTEX_REQUEST_TYPES = {"shared", "dedicated"}


def _prepend_user_context(llm_request, text: str) -> None:
    """Insert dynamic context as a user content at the start of contents.

    By placing dynamic per-request information (temporal context, corrections,
    memories, examples) in user content instead of system_instruction, we keep
    the system instruction stable across requests.  This enables Gemini context
    caching — both implicit (automatic prefix dedup) and explicit (via ADK
    ContextCacheConfig) — since the cache fingerprint depends on a stable
    system_instruction.
    """
    llm_request.contents.insert(
        0,
        types.Content(role="user", parts=[types.Part(text=text)]),
    )


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


_HISTORICAL_HINT_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b(histor|regulament|regulation|seasons?|championship)\b",
    re.I,
)


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


def _resolve_vertex_request_type() -> str:
    raw = os.environ.get("F1_VERTEX_LLM_REQUEST_TYPE", _DEFAULT_VERTEX_REQUEST_TYPE)
    value = str(raw).strip().lower()
    if value in _VALID_VERTEX_REQUEST_TYPES:
        return value

    logger.warning(
        "Invalid F1_VERTEX_LLM_REQUEST_TYPE=%r; falling back to %s",
        raw,
        _DEFAULT_VERTEX_REQUEST_TYPE,
    )
    return _DEFAULT_VERTEX_REQUEST_TYPE


def apply_throughput_request_type(callback_context, llm_request):
    """Before-model callback: route request as shared or dedicated throughput.

    Uses Vertex header ``X-Vertex-AI-LLM-Request-Type``:
    - ``shared`` -> Dynamic Shared Quota (pay-as-you-go)
    - ``dedicated`` -> Provisioned Throughput
    """
    del callback_context  # unused

    request_type = _resolve_vertex_request_type()
    if not llm_request.config:
        llm_request.config = types.GenerateContentConfig()
    if not llm_request.config.http_options:
        llm_request.config.http_options = types.HttpOptions()

    headers = dict(llm_request.config.http_options.headers or {})
    headers["X-Vertex-AI-LLM-Request-Type"] = request_type
    llm_request.config.http_options.headers = headers

    logger.info(
        "throughput_route | request_type=%s model=%s", request_type, llm_request.model
    )
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

    query_type = _classify_cache_query(user_text)

    # Time-sensitive queries should prefer fresh tool calls.
    if _query_requires_web_data(user_text):
        logger.info(
            "semantic_cache | query_type=%s outcome=bypass lookup_ms=0.00 "
            "candidates=0 top1=-1.0000 evicted=0",
            query_type,
        )
        return None

    if hasattr(cache, "lookup"):
        result = cache.lookup(user_text)
        hit = result.answer
        outcome = result.outcome
        lookup_ms = result.lookup_ms
        candidates = result.candidates_scanned
        top1 = result.similarity_top1 if result.similarity_top1 is not None else -1.0
        evicted = result.evicted_count
    else:
        hit = cache.get(user_text)
        outcome = "hit" if hit is not None else "miss"
        lookup_ms = 0.0
        candidates = 0
        top1 = -1.0
        evicted = 0

    logger.info(
        "semantic_cache | query_type=%s outcome=%s lookup_ms=%.2f "
        "candidates=%d top1=%.4f evicted=%d",
        query_type,
        outcome,
        lookup_ms,
        candidates,
        top1,
        evicted,
    )

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
_MEMORY_GENERATE_FLAG_KEY = "f1_generate_memory"

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


def _extract_user_id(callback_context) -> str | None:
    user_id = getattr(callback_context, "user_id", None)
    if isinstance(user_id, str) and user_id.strip():
        return user_id.strip()

    invocation_context = getattr(callback_context, "invocation_context", None)
    nested_user_id = getattr(invocation_context, "user_id", None)
    if isinstance(nested_user_id, str) and nested_user_id.strip():
        return nested_user_id.strip()

    state = getattr(callback_context, "state", None)
    if isinstance(state, dict):
        fallback = state.get("user_id")
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()

    return None


def _extract_session_name(callback_context) -> str | None:
    session_name = getattr(callback_context, "session_name", None)
    if isinstance(session_name, str) and "/sessions/" in session_name:
        return session_name

    session = getattr(callback_context, "session", None)
    session_from_obj = getattr(session, "name", None)
    if isinstance(session_from_obj, str) and "/sessions/" in session_from_obj:
        return session_from_obj

    session_id = getattr(callback_context, "session_id", None)
    if not isinstance(session_id, str) or not session_id.strip():
        return None

    settings = load_memory_bank_settings()
    if (
        not settings.project_id
        or not settings.location
        or not settings.agent_engine_name
    ):
        return None

    if "/sessions/" in session_id:
        return session_id

    return f"{settings.agent_engine_name}/sessions/{session_id.strip()}"


def _mark_memory_generation(callback_context) -> None:
    state = getattr(callback_context, "state", None)
    if isinstance(state, dict):
        state[_MEMORY_GENERATE_FLAG_KEY] = True


def _consume_memory_generation_flag(callback_context) -> bool:
    state = getattr(callback_context, "state", None)
    if not isinstance(state, dict):
        return False

    flagged = bool(state.get(_MEMORY_GENERATE_FLAG_KEY, False))
    state.pop(_MEMORY_GENERATE_FLAG_KEY, None)
    return flagged


def inject_corrections(callback_context, llm_request):
    """Before-model callback: inject stored corrections into the prompt.

    If the user has previously corrected the agent in this session,
    prepend those corrections as user content so the model is aware of
    them without mutating the system instruction (preserving cache
    fingerprint stability).
    """
    corrections = _get_corrections(callback_context)
    if not corrections:
        return None

    corrections_text = "\n".join(f"- {c}" for c in corrections)
    addendum = (
        "## User corrections from this session\n"
        "The user has corrected you on the following points. "
        "Take these into account and do NOT repeat the same mistakes:\n"
        f"{corrections_text}"
    )

    _prepend_user_context(llm_request, addendum)
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

    _prepend_user_context(llm_request, addendum)
    logger.info(
        "Injected dynamic examples: count=%s top_similarity=%s",
        metadata.get("example_count"),
        metadata.get("top_similarity"),
    )
    return None


def inject_long_term_memories(callback_context, llm_request):
    """Before-model callback: inject relevant cross-session memories."""
    user_id = _extract_user_id(callback_context)
    if not user_id:
        return None

    user_text = _extract_user_text(callback_context, llm_request)
    addendum, metadata = build_memory_addendum(user_id=user_id, query_text=user_text)
    if not addendum:
        logger.debug(
            "Memory injection skipped (enabled=%s configured=%s)",
            metadata.get("enabled"),
            metadata.get("configured"),
        )
        return None

    _prepend_user_context(llm_request, addendum)
    logger.info("Injected long-term memories: count=%s", metadata.get("memory_count"))
    return None


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
    _mark_memory_generation(callback_context)
    logger.info("Correction stored: %s", correction[:80])

    return None  # don't modify the response


def sync_memory_bank(callback_context, llm_response):
    """After-model callback: trigger long-term memory generation from session."""
    del llm_response  # unused

    settings = load_memory_bank_settings()
    if not settings.enabled:
        return None

    should_generate = _consume_memory_generation_flag(callback_context)

    if not should_generate and settings.generate_on_correction_only:
        return None

    session_name = _extract_session_name(callback_context)
    if not session_name:
        logger.debug("Memory generation skipped: missing session name")
        return None

    user_id = _extract_user_id(callback_context)
    generated = generate_memories_from_session(
        session_name=session_name,
        user_id=user_id,
        wait_for_completion=False,
    )
    if generated:
        logger.info("Triggered memory generation for session: %s", session_name)

    return None


def log_context_cache_metrics(callback_context, llm_response):
    """After-model callback: log Gemini context cache token metrics.

    Inspects ``usage_metadata`` on the response to report how many prompt
    tokens were served from cache vs total, enabling monitoring of cache
    effectiveness.
    """
    del callback_context  # unused
    if not llm_response:
        return None

    usage = getattr(llm_response, "usage_metadata", None)
    if not usage:
        return None

    cached = getattr(usage, "cached_content_token_count", 0) or 0
    total = getattr(usage, "prompt_token_count", 0) or 0

    if total > 0:
        logger.info(
            "context_cache | prompt_tokens=%d cached_tokens=%d ratio=%.2f",
            total,
            cached,
            cached / total,
        )

    return None


def preflight_token_check(callback_context, llm_request):
    """Before-model callback: count tokens and truncate if over budget.

    Runs as the **last** before-model callback, after all context injection
    and model routing.  Calls the Gemini CountTokens API to verify the
    request fits within the configured threshold.  If it exceeds the limit,
    progressively removes injected context blocks (examples → memories →
    corrections → temporal) until the request is within budget.

    Gracefully degrades: if CountTokens fails, logs a warning and proceeds
    without truncation.
    """
    del callback_context  # unused
    try:
        _check_and_truncate(llm_request)
    except Exception:  # noqa: BLE001
        logger.warning(
            "preflight_token_check failed — proceeding without truncation",
            exc_info=True,
        )
    return None
