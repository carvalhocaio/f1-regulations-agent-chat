"""Before/after-model callbacks for grounding policy enforcement."""

from __future__ import annotations

import json
import logging
import os

from google.genai import types

from f1_agent.cb_helpers import _extract_user_text, _prepend_user_context
from f1_agent.cb_temporal import _query_requires_web_data
from f1_agent.env_utils import env_bool
from f1_agent.response_contract import (
    CONTRACT_ID_SOURCES_BLOCK_V1,
)

logger = logging.getLogger(__name__)

_GROUNDING_POLICY_ENABLED_ENV = "F1_GROUNDING_POLICY_ENABLED"
_GROUNDING_POLICY_MODE_ENV = "F1_GROUNDING_POLICY_MODE"
_GROUNDING_TIME_SENSITIVE_SOURCE_ENV = "F1_GROUNDING_TIME_SENSITIVE_SOURCE"
_DEFAULT_GROUNDING_POLICY_MODE = "observe"
_VALID_GROUNDING_POLICY_MODES = {"observe", "enforce"}
_DEFAULT_GROUNDING_TIME_SENSITIVE_SOURCE = "google"
_VALID_GROUNDING_TIME_SENSITIVE_SOURCES = {"google"}
_GROUNDING_POLICY_STATE_KEY = "f1_grounding_policy"
_GROUNDING_REQUIRED_STATE_KEY = "f1_grounding_required"
_GROUNDING_SOURCE_STATE_KEY = "f1_grounding_source"
_AUTO_GROUNDING_CONTRACT_STATE_KEY = "f1_auto_grounding_contract"
_GROUNDING_VALIDATION_COUNTERS: dict[str, int] = {}


def _resolve_grounding_policy_mode() -> str:
    raw = os.environ.get(_GROUNDING_POLICY_MODE_ENV, _DEFAULT_GROUNDING_POLICY_MODE)
    value = str(raw).strip().lower()
    if value in _VALID_GROUNDING_POLICY_MODES:
        return value
    logger.warning(
        "Invalid %s=%r; falling back to %s",
        _GROUNDING_POLICY_MODE_ENV,
        raw,
        _DEFAULT_GROUNDING_POLICY_MODE,
    )
    return _DEFAULT_GROUNDING_POLICY_MODE


def _resolve_time_sensitive_grounding_source() -> str:
    raw = os.environ.get(
        _GROUNDING_TIME_SENSITIVE_SOURCE_ENV, _DEFAULT_GROUNDING_TIME_SENSITIVE_SOURCE
    )
    value = str(raw).strip().lower()
    if value in _VALID_GROUNDING_TIME_SENSITIVE_SOURCES:
        return value
    logger.warning(
        "Invalid %s=%r; falling back to %s",
        _GROUNDING_TIME_SENSITIVE_SOURCE_ENV,
        raw,
        _DEFAULT_GROUNDING_TIME_SENSITIVE_SOURCE,
    )
    return _DEFAULT_GROUNDING_TIME_SENSITIVE_SOURCE


def _bump_grounding_validation_counter(policy: str, outcome: str) -> None:
    key = f"{policy}:{outcome}"
    _GROUNDING_VALIDATION_COUNTERS[key] = _GROUNDING_VALIDATION_COUNTERS.get(key, 0) + 1


def get_grounding_validation_counters() -> dict[str, int]:
    """Return a snapshot of grounding validation counters."""
    return dict(_GROUNDING_VALIDATION_COUNTERS)


def _extract_response_contract_id_from_state(callback_context) -> str | None:
    """Check if a response contract is already set in state."""
    state = getattr(callback_context, "state", None)
    if isinstance(state, dict):
        for key in ("response_contract_id", "f1_response_contract_id"):
            value = state.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def apply_grounding_policy(callback_context, llm_request):
    """Before-model callback: apply grounding policy for factual-critical routes.

    Phase 1 policy (recommended): enforce Google Search grounding for time-sensitive
    questions ("now/current/latest/live", post-2024 ranges, recency requests).
    """
    if not env_bool(_GROUNDING_POLICY_ENABLED_ENV, True):
        return None

    user_text = _extract_user_text(callback_context, llm_request)
    if not user_text:
        return None

    state = getattr(callback_context, "state", None)
    requires_grounding = _query_requires_web_data(user_text)
    policy = "time_sensitive_public" if requires_grounding else "non_critical"
    source = (
        _resolve_time_sensitive_grounding_source() if requires_grounding else "none"
    )

    if isinstance(state, dict):
        state[_GROUNDING_POLICY_STATE_KEY] = policy
        state[_GROUNDING_REQUIRED_STATE_KEY] = requires_grounding
        state[_GROUNDING_SOURCE_STATE_KEY] = source

    if not requires_grounding:
        if (
            isinstance(state, dict)
            and state.get(_AUTO_GROUNDING_CONTRACT_STATE_KEY)
            and state.get("f1_response_contract_id") == CONTRACT_ID_SOURCES_BLOCK_V1
        ):
            state.pop("f1_response_contract_id", None)
            state.pop(_AUTO_GROUNDING_CONTRACT_STATE_KEY, None)
        return None

    addendum = (
        "## Grounding policy (factual-critical route)\n"
        "This request is time-sensitive/current-state. You MUST ground claims with "
        "fresh evidence before finalizing the answer.\n"
        "Requirements:\n"
        "- Do not answer from stale memory for current-state facts.\n"
        "- Include at least one web source with URL in the final response.\n"
        "- If grounding evidence is insufficient, say you could not verify confidently "
        "and ask for clarification."
    )
    _prepend_user_context(llm_request, addendum)

    if isinstance(state, dict) and not _extract_response_contract_id_from_state(
        callback_context
    ):
        state["f1_response_contract_id"] = CONTRACT_ID_SOURCES_BLOCK_V1
        state[_AUTO_GROUNDING_CONTRACT_STATE_KEY] = True

    logger.info(
        "grounding_policy | policy=%s required=%s source=%s",
        policy,
        requires_grounding,
        source,
    )
    return None


# ── After-model grounding validation ─────────────────────────────────────


def _response_contains_grounding_metadata(llm_response) -> bool:
    if not llm_response:
        return False

    candidates = getattr(llm_response, "candidates", None)
    if isinstance(candidates, list):
        for candidate in candidates:
            if getattr(candidate, "grounding_metadata", None) is not None:
                return True

    grounding_metadata = getattr(llm_response, "grounding_metadata", None)
    return grounding_metadata is not None


def _extract_web_sources_count_from_response_text(response_text: str) -> int:
    if not response_text:
        return 0

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return 0

    if not isinstance(payload, dict):
        return 0

    sources = payload.get("sources")
    if not isinstance(sources, list):
        return 0

    return sum(
        1
        for source in sources
        if isinstance(source, dict) and source.get("source_type") == "web"
    )


def _extract_response_text(llm_response) -> str:
    if not llm_response or not llm_response.content or not llm_response.content.parts:
        return ""
    texts = [part.text for part in llm_response.content.parts if part.text]
    return "\n".join(texts).strip()


def validate_grounding_outcome(callback_context, llm_response):
    """After-model callback: verify grounding evidence for critical routes."""
    if not env_bool(_GROUNDING_POLICY_ENABLED_ENV, True):
        return None

    state = getattr(callback_context, "state", None)
    required = False
    policy = "unknown"
    source = "unknown"
    if isinstance(state, dict):
        required = bool(state.get(_GROUNDING_REQUIRED_STATE_KEY, False))
        policy = str(state.get(_GROUNDING_POLICY_STATE_KEY, "unknown"))
        source = str(state.get(_GROUNDING_SOURCE_STATE_KEY, "unknown"))

    if not required:
        return None

    response_text = _extract_response_text(llm_response)
    web_sources_count = _extract_web_sources_count_from_response_text(response_text)
    metadata_present = _response_contains_grounding_metadata(llm_response)
    grounded = web_sources_count > 0 or metadata_present
    mode = _resolve_grounding_policy_mode()
    outcome = "success" if grounded else "missing"
    _bump_grounding_validation_counter(policy, outcome)

    logger.info(
        "grounding_validation | policy=%s required=%s source=%s outcome=%s "
        "web_sources=%d metadata_present=%s mode=%s",
        policy,
        required,
        source,
        outcome,
        web_sources_count,
        metadata_present,
        mode,
    )

    if grounded or mode != "enforce":
        return None

    fallback_payload = {
        "schema_version": "v1",
        "answer": (
            "I could not confidently verify this time-sensitive answer with fresh "
            "grounded web evidence in this attempt. Please retry or refine the query."
        ),
        "sources": [],
    }
    if llm_response and llm_response.content:
        llm_response.content.parts = [
            types.Part(text=json.dumps(fallback_payload, ensure_ascii=True))
        ]
    return None
