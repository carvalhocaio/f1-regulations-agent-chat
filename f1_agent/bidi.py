"""Client-side helpers for protocolized bidirectional streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from f1_agent.streaming_protocol import (
    EVENT_DELTA,
    EVENT_ERROR,
    EVENT_TURN_END,
    EVENT_TURN_START,
    build_stream_event,
    extract_bidi_text,
    is_bidi_end_of_turn,
)


async def iter_bidi_turn_events(
    *,
    connection: Any,
    request: Mapping[str, Any],
    request_id: str,
    user_id: str | None,
    session_id: str | None,
    receive_timeout_seconds: float = 60.0,
) -> AsyncGenerator[dict[str, Any], None]:
    """Send one request and stream protocolized events until turn end."""
    sequence = 1
    yield build_stream_event(
        event_type=EVENT_TURN_START,
        request_id=request_id,
        sequence=sequence,
        user_id=user_id,
        session_id=session_id,
        payload={"input": dict(request)},
    )

    await connection.send(dict(request))

    while True:
        try:
            raw = await asyncio.wait_for(
                connection.receive(), timeout=max(0.1, receive_timeout_seconds)
            )
        except Exception as exc:
            sequence += 1
            yield build_stream_event(
                event_type=EVENT_ERROR,
                request_id=request_id,
                sequence=sequence,
                user_id=user_id,
                session_id=session_id,
                payload={
                    "error": f"{type(exc).__name__}: {exc}",
                    "recoverable": True,
                },
            )
            break

        if not isinstance(raw, Mapping):
            sequence += 1
            yield build_stream_event(
                event_type=EVENT_DELTA,
                request_id=request_id,
                sequence=sequence,
                user_id=user_id,
                session_id=session_id,
                payload={"raw": raw},
            )
            continue

        text = extract_bidi_text(raw)
        if text is not None:
            sequence += 1
            yield build_stream_event(
                event_type=EVENT_DELTA,
                request_id=request_id,
                sequence=sequence,
                user_id=user_id,
                session_id=session_id,
                payload={"text": text, "raw": dict(raw)},
            )

        if is_bidi_end_of_turn(raw):
            sequence += 1
            yield build_stream_event(
                event_type=EVENT_TURN_END,
                request_id=request_id,
                sequence=sequence,
                user_id=user_id,
                session_id=session_id,
                payload={"reason": "end_of_turn", "raw": dict(raw)},
            )
            break
