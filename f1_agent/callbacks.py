"""
ADK callbacks for model routing and semantic caching.

- ``route_model``: before-model callback that classifies the user question
  and switches to Gemini Flash for simple queries (keeping Pro for complex ones).
- ``check_cache`` / ``store_cache``: before/after-model callbacks for the
  semantic answer cache.
"""

from __future__ import annotations

import logging
import re

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from f1_agent.cache import SemanticCache

logger = logging.getLogger(__name__)

# ── Model routing ───────────────────────────────────────────────────────

FLASH_MODEL = "gemini-2.5-flash"

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
