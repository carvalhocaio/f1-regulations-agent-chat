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

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from f1_agent.cache import SemanticCache

logger = logging.getLogger(__name__)

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
            # Detect if answer used google_search (short TTL)
            used_web = "🌐" in answer or "google_search" in answer.lower()
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
