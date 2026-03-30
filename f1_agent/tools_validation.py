"""Shared validation helpers and error formatting for agent tools."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from f1_agent.tool_metrics import emit_tool_validation_error_metric

logger = logging.getLogger(__name__)

_MAX_QUERY_LEN = 500
_TOOL_VALIDATION_ERROR_COUNTER: Counter[str] = Counter()


def get_tool_validation_error_counters() -> dict[str, int]:
    """Return a snapshot of validation error counters by tool/code."""
    return dict(_TOOL_VALIDATION_ERROR_COUNTER)


def _normalize_non_empty_text(
    *, value: object, field_name: str, max_len: int
) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > max_len:
        logger.warning(
            "Truncating %s from %d to %d chars", field_name, len(normalized), max_len
        )
        normalized = normalized[:max_len]
    return normalized


def _tool_error(
    *,
    tool_name: str,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counter_key = f"{tool_name}:{code}"
    _TOOL_VALIDATION_ERROR_COUNTER[counter_key] += 1
    logger.warning(
        "tool_validation_error | tool=%s code=%s count=%d message=%s",
        tool_name,
        code,
        _TOOL_VALIDATION_ERROR_COUNTER[counter_key],
        message,
    )
    emit_tool_validation_error_metric(tool_name=tool_name, error_code=code)

    payload: dict[str, Any] = {
        "status": "error",
        "message": message,
        "error": {"tool_name": tool_name, "code": code, "message": message},
    }
    if details:
        payload["error"]["details"] = details
    return payload
