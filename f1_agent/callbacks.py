"""Runtime callbacks for model routing, semantic caching, and session corrections.

This module is a re-export facade. Implementations live in focused submodules
(``cb_helpers``, ``cb_model_routing``, ``cb_temporal``, etc.) for single-
responsibility compliance. All public and private names that were historically
importable from ``f1_agent.callbacks`` are re-exported here for backward
compatibility.
"""

from __future__ import annotations

import logging

from f1_agent.cb_corrections import (  # noqa: F401
    _CORRECTIONS_KEY,
    _get_corrections,
    _is_correction,
    _store_correction,
    detect_corrections,
    inject_corrections,
)
from f1_agent.cb_grounding import (  # noqa: F401
    apply_grounding_policy,
    get_grounding_validation_counters,
    validate_grounding_outcome,
)
from f1_agent.cb_helpers import (  # noqa: F401
    _DB_MAX_YEAR,
    _current_date,
    _current_year,
    _extract_user_text,
    _prepend_user_context,
)
from f1_agent.cb_model_routing import (  # noqa: F401
    FLASH_MODEL,
    _is_complex_question,
    _is_pro_quota_exhausted,
    apply_throughput_request_type,
    route_model,
)
from f1_agent.cb_response_validation import (  # noqa: F401
    apply_response_contract,
    get_structured_response_validation_counters,
    validate_structured_response,
)
from f1_agent.cb_semantic_cache import (  # noqa: F401
    _get_cache,
    check_cache,
    store_cache,
)
from f1_agent.cb_temporal import (  # noqa: F401
    _classify_cache_query,
    _query_requires_web_data,
    _resolve_temporal_references,
    _runtime_temporal_addendum,
    inject_runtime_temporal_context,
)
from f1_agent.example_store import build_dynamic_examples_addendum
from f1_agent.token_preflight import check_and_truncate as _check_and_truncate

logger = logging.getLogger(__name__)


# ── Small inline callbacks (not worth their own module) ──────────────────


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


def preflight_token_check(callback_context, llm_request):
    """Before-model callback: count tokens and truncate if over budget.

    Runs as the **last** before-model callback, after all context injection
    and model routing.  Calls the Gemini CountTokens API to verify the
    request fits within the configured threshold. If it exceeds the limit,
    progressively removes injected context blocks (examples → corrections →
    temporal) until the request is within budget.

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
