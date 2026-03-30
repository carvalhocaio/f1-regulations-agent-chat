"""Before/after-model callbacks for structured response contract enforcement."""

from __future__ import annotations

import json
import logging

from google.genai import types

from f1_agent.env_utils import env_bool
from f1_agent.response_contract import (
    CONTRACT_ID_COMPARISON_TABLE_V1,
    CONTRACT_ID_SOURCES_BLOCK_V1,
    get_response_contract,
    list_response_contract_ids,
    validate_contract_payload,
)

logger = logging.getLogger(__name__)

_STRUCTURED_RESPONSE_ENABLED_ENV = "F1_STRUCTURED_RESPONSE_ENABLED"
_RESPONSE_CONTRACT_STATE_KEYS = (
    "response_contract_id",
    "f1_response_contract_id",
)
_ACTIVE_RESPONSE_CONTRACT_KEY = "f1_active_response_contract_id"
_STRUCTURED_RESPONSE_VALIDATION_COUNTERS: dict[str, int] = {}


def _extract_response_contract_id(callback_context) -> str | None:
    state = getattr(callback_context, "state", None)
    if isinstance(state, dict):
        for key in _RESPONSE_CONTRACT_STATE_KEYS:
            value = state.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    invocation_context = getattr(callback_context, "invocation_context", None)
    for key in _RESPONSE_CONTRACT_STATE_KEYS:
        value = getattr(invocation_context, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = getattr(invocation_context, "metadata", None)
    if isinstance(metadata, dict):
        for key in _RESPONSE_CONTRACT_STATE_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for attr in ("request", "request_payload", "payload"):
        payload = getattr(invocation_context, attr, None)
        if isinstance(payload, dict):
            for key in _RESPONSE_CONTRACT_STATE_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    return None


def apply_response_contract(callback_context, llm_request):
    """Before-model callback: apply strict response schema for critical routes."""
    state = getattr(callback_context, "state", None)
    if not env_bool(_STRUCTURED_RESPONSE_ENABLED_ENV, True):
        if isinstance(state, dict):
            state.pop(_ACTIVE_RESPONSE_CONTRACT_KEY, None)
        return None

    contract_id = _extract_response_contract_id(callback_context)
    if not contract_id:
        if isinstance(state, dict):
            state.pop(_ACTIVE_RESPONSE_CONTRACT_KEY, None)
        return None

    contract = get_response_contract(contract_id)
    if not contract:
        logger.warning(
            "structured_response | unknown_contract_id=%s supported=%s",
            contract_id,
            ",".join(list_response_contract_ids()),
        )
        if isinstance(state, dict):
            state.pop(_ACTIVE_RESPONSE_CONTRACT_KEY, None)
        return None

    if not llm_request.config:
        llm_request.config = types.GenerateContentConfig()

    llm_request.config.response_mime_type = contract.response_mime_type
    llm_request.config.response_schema = contract.response_schema
    if isinstance(state, dict):
        state[_ACTIVE_RESPONSE_CONTRACT_KEY] = contract_id

    logger.info("structured_response | contract_id=%s", contract_id)
    return None


def get_structured_response_validation_counters() -> dict[str, int]:
    """Return a snapshot of structured response validation counters."""
    return dict(_STRUCTURED_RESPONSE_VALIDATION_COUNTERS)


def _bump_structured_response_counter(contract_id: str, outcome: str) -> None:
    key = f"{contract_id}:{outcome}"
    _STRUCTURED_RESPONSE_VALIDATION_COUNTERS[key] = (
        _STRUCTURED_RESPONSE_VALIDATION_COUNTERS.get(key, 0) + 1
    )


def _extract_response_text(llm_response) -> str:
    if not llm_response or not llm_response.content or not llm_response.content.parts:
        return ""
    texts = [part.text for part in llm_response.content.parts if part.text]
    return "\n".join(texts).strip()


def _build_structured_fallback_payload(
    contract_id: str, raw_text: str, reason: str
) -> dict[str, object]:
    fallback_excerpt = (raw_text or "").strip()
    if len(fallback_excerpt) > 500:
        fallback_excerpt = fallback_excerpt[:500]

    if contract_id == CONTRACT_ID_SOURCES_BLOCK_V1:
        return {
            "schema_version": "v1",
            "answer": fallback_excerpt
            or "Structured output fallback applied due to response validation failure.",
            "sources": [],
        }

    if contract_id == CONTRACT_ID_COMPARISON_TABLE_V1:
        note_text = reason
        if fallback_excerpt:
            note_text = f"{reason}: {fallback_excerpt}"
        return {
            "schema_version": "v1",
            "title": "Structured output fallback",
            "columns": ["note"],
            "rows": [[note_text]],
            "notes": ["Generated fallback payload to keep parser compatibility."],
        }

    return {
        "schema_version": "v1",
        "message": "Structured output fallback applied.",
        "reason": reason,
    }


def _get_active_response_contract_id(callback_context) -> str | None:
    state = getattr(callback_context, "state", None)
    if isinstance(state, dict):
        value = state.get(_ACTIVE_RESPONSE_CONTRACT_KEY)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _extract_response_contract_id(callback_context)


def validate_structured_response(callback_context, llm_response):
    """After-model callback: validate/enforce structured response payloads."""
    if not env_bool(_STRUCTURED_RESPONSE_ENABLED_ENV, True):
        return None

    contract_id = _get_active_response_contract_id(callback_context)
    if not contract_id:
        return None

    state = getattr(callback_context, "state", None)
    if isinstance(state, dict):
        state.pop(_ACTIVE_RESPONSE_CONTRACT_KEY, None)

    response_text = _extract_response_text(llm_response)
    if not response_text:
        reason = "empty_response_text"
        fallback_payload = _build_structured_fallback_payload(contract_id, "", reason)
        _bump_structured_response_counter(contract_id, "parse_failure")
        if llm_response and llm_response.content:
            llm_response.content.parts = [
                types.Part(text=json.dumps(fallback_payload, ensure_ascii=True))
            ]
        logger.warning(
            "structured_response_validation | contract_id=%s outcome=parse_failure reason=%s",
            contract_id,
            reason,
        )
        return None

    try:
        parsed_payload = json.loads(response_text)
    except json.JSONDecodeError:
        reason = "invalid_json"
        fallback_payload = _build_structured_fallback_payload(
            contract_id, response_text, reason
        )
        _bump_structured_response_counter(contract_id, "parse_failure")
        if llm_response and llm_response.content:
            llm_response.content.parts = [
                types.Part(text=json.dumps(fallback_payload, ensure_ascii=True))
            ]
        logger.warning(
            "structured_response_validation | contract_id=%s outcome=parse_failure reason=%s",
            contract_id,
            reason,
        )
        return None

    is_valid, error = validate_contract_payload(contract_id, parsed_payload)
    if is_valid:
        _bump_structured_response_counter(contract_id, "success")
        logger.info(
            "structured_response_validation | contract_id=%s outcome=success",
            contract_id,
        )
        return None

    fallback_payload = _build_structured_fallback_payload(
        contract_id,
        response_text,
        error or "schema_validation_error",
    )
    _bump_structured_response_counter(contract_id, "schema_failure")
    if llm_response and llm_response.content:
        llm_response.content.parts = [
            types.Part(text=json.dumps(fallback_payload, ensure_ascii=True))
        ]
    logger.warning(
        "structured_response_validation | contract_id=%s outcome=schema_failure error=%s",
        contract_id,
        error,
    )
    return None
