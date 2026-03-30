"""Before/after-model callbacks for semantic answer caching."""

from __future__ import annotations

import logging

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from f1_agent.cache import SemanticCache
from f1_agent.cb_helpers import _extract_user_text
from f1_agent.cb_temporal import _classify_cache_query, _query_requires_web_data

logger = logging.getLogger(__name__)

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

    result = cache.lookup(user_text)
    hit = result.answer
    outcome = result.outcome
    lookup_ms = result.lookup_ms
    candidates = result.candidates_scanned
    top1 = result.similarity_top1 if result.similarity_top1 is not None else -1.0
    evicted = result.evicted_count

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
            used_web = "🌐" in answer or _query_requires_web_data(user_text)
            cache.put(user_text, answer, web_source=used_web)
            logger.debug("Cache STORE for: %s", user_text[:80])

    return None  # don't modify the response
