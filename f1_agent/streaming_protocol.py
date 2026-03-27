"""Protocol helpers for interactive streaming events.

This module defines a stable envelope (`stream_protocol_version=v1`) used by
interactive clients, independent of the raw SDK event shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

STREAM_PROTOCOL_VERSION = "v1"

EVENT_TURN_START = "turn_start"
EVENT_DELTA = "delta"
EVENT_TOOL_STATUS = "tool_status"
EVENT_TURN_END = "turn_end"
EVENT_ERROR = "error"

_VALID_EVENT_TYPES = {
    EVENT_TURN_START,
    EVENT_DELTA,
    EVENT_TOOL_STATUS,
    EVENT_TURN_END,
    EVENT_ERROR,
}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_stream_event(
    *,
    event_type: str,
    request_id: str,
    sequence: int,
    user_id: str | None,
    session_id: str | None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a protocol v1 stream event envelope."""
    if event_type not in _VALID_EVENT_TYPES:
        raise ValueError(f"Unsupported stream event_type: {event_type}")

    return {
        "stream_protocol_version": STREAM_PROTOCOL_VERSION,
        "event_type": event_type,
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "sequence": sequence,
        "ts": utc_now_iso(),
        "payload": dict(payload or {}),
    }


def is_bidi_end_of_turn(raw_event: Mapping[str, Any]) -> bool:
    """Return true when event indicates end of turn.

    Agent Engine bidi docs show `response["bidiStreamOutput"]["output"] ==
    "end of turn"` as the sentinel for turn completion.
    """
    bidi_output = raw_event.get("bidiStreamOutput")
    if not isinstance(bidi_output, Mapping):
        return False

    output = bidi_output.get("output")
    if not isinstance(output, str):
        return False

    normalized = output.strip().lower()
    return normalized == "end of turn"


def extract_bidi_text(raw_event: Mapping[str, Any]) -> str | None:
    """Extract incremental text-like content from a raw bidi event."""
    bidi_output = raw_event.get("bidiStreamOutput")
    if not isinstance(bidi_output, Mapping):
        return None

    output = bidi_output.get("output")
    if not isinstance(output, str):
        return None

    normalized = output.strip()
    if not normalized:
        return None
    if normalized.lower() == "end of turn":
        return None
    return output
