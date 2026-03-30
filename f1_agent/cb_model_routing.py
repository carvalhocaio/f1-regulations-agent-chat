"""Before-model callbacks for model routing and throughput configuration."""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime

from google.genai import types

from f1_agent.cb_helpers import _extract_user_text

logger = logging.getLogger(__name__)

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


# ── Pro quota fallback ───────────────────────────────────────────────────

_PRO_QUOTA_STATE_KEY = "f1_pro_quota_exhausted_at"
_PRO_QUOTA_FALLBACK_HOURS = 6


def _is_pro_quota_exhausted(callback_context) -> bool:
    """Check if Pro quota was recently exhausted (within fallback window)."""
    state = getattr(callback_context, "state", None)
    if not isinstance(state, dict):
        return False
    exhausted_at = state.get(_PRO_QUOTA_STATE_KEY)
    if not exhausted_at:
        return False
    try:
        ts = datetime.fromisoformat(exhausted_at)
        elapsed_hours = (
            datetime.now(UTC)
            - ts.replace(tzinfo=UTC if ts.tzinfo is None else ts.tzinfo)
        ).total_seconds() / 3600
        if elapsed_hours > _PRO_QUOTA_FALLBACK_HOURS:
            state.pop(_PRO_QUOTA_STATE_KEY, None)
            return False
        return True
    except (ValueError, TypeError):
        return False


def route_model(callback_context, llm_request):
    """Before-model callback: route queries to appropriate model.

    Routes to Flash when:
    - The query is simple (not complex)
    - Pro quota has been recently exhausted (automatic fallback)
    """
    user_text = _extract_user_text(callback_context, llm_request)
    if not user_text:
        return None

    if _is_pro_quota_exhausted(callback_context):
        logger.info("Routing to Flash (Pro quota exhausted): %s", user_text[:80])
        llm_request.model = FLASH_MODEL
        return None

    if _is_complex_question(user_text):
        logger.debug("Routing to Pro (complex): %s", user_text[:80])
        return None  # keep the default model (Pro)

    logger.debug("Routing to Flash (simple): %s", user_text[:80])
    llm_request.model = FLASH_MODEL
    return None


# ── Throughput request type ──────────────────────────────────────────────

_DEFAULT_VERTEX_REQUEST_TYPE = "shared"
_VALID_VERTEX_REQUEST_TYPES = {"shared", "dedicated"}


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
