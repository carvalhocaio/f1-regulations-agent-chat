"""WebSocket bridge for interactive bidi streaming.

This module is framework-agnostic. The ``websocket`` object is expected to
provide ``receive_json()``, ``send_json(payload)``, and optional ``close()``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from f1_agent.bidi import iter_bidi_turn_events
from f1_agent.streaming_protocol import EVENT_ERROR, EVENT_TURN_END, build_stream_event

ConnectionFactory = Callable[[], AbstractAsyncContextManager[Any]]


def _extract_text_from_stream_event(event: Any) -> str:
    if isinstance(event, str):
        return event
    if isinstance(event, dict):
        for key in ("text", "output", "delta"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(event)
    return str(event)


class _AsyncStreamFallbackConnection:
    """Adapter that mimics bidi connection using `async_stream_query`."""

    def __init__(self, remote_agent: Any):
        self._remote_agent = remote_agent
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._producer_task: asyncio.Task[None] | None = None

    async def send(self, payload: dict[str, Any]) -> None:
        message = str(payload.get("input", "")).strip()
        user_id = str(payload.get("user_id") or payload.get("userId") or "ws-user")
        if not message:
            raise ValueError("payload.input is required")

        async def _produce() -> None:
            try:
                async for event in self._remote_agent.async_stream_query(
                    user_id=user_id, message=message
                ):
                    text = _extract_text_from_stream_event(event)
                    await self._queue.put({"bidiStreamOutput": {"output": text}})
                await self._queue.put({"bidiStreamOutput": {"output": "end of turn"}})
            except Exception as exc:
                await self._queue.put(exc)

        self._producer_task = asyncio.create_task(_produce())

    async def receive(self) -> Any:
        item = await self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item


def _has_live_agent_engine_connect(client: Any) -> bool:
    aio = getattr(client, "aio", None)
    live = getattr(aio, "live", None)
    agent_engines = getattr(live, "agent_engines", None)
    connect = getattr(agent_engines, "connect", None)
    return callable(connect)


def build_agent_connection_factory(
    *,
    client: Any,
    agent_engine: str,
    remote_agent: Any | None = None,
    class_method: str = "bidi_stream_query",
) -> ConnectionFactory:
    """Build best-available connection factory for streaming turns.

    Preference order:
    1) Native live bidi connection (SDK with `aio.live.agent_engines.connect`).
    2) Fallback adapter backed by `remote_agent.async_stream_query`.
    """
    if _has_live_agent_engine_connect(client):

        @asynccontextmanager
        async def _live_factory():
            async with client.aio.live.agent_engines.connect(
                agent_engine=agent_engine,
                config={"class_method": class_method},
            ) as connection:
                yield connection

        return _live_factory

    if remote_agent is None or not hasattr(remote_agent, "async_stream_query"):
        raise RuntimeError(
            "No native live agent connection support and no async_stream fallback "
            "available"
        )

    @asynccontextmanager
    async def _fallback_factory():
        yield _AsyncStreamFallbackConnection(remote_agent)

    return _fallback_factory



@dataclass
class _ActiveTurn:
    request_id: str
    user_id: str | None
    session_id: str | None
    task: asyncio.Task[None]


async def websocket_bidi_loop(
    *,
    websocket: Any,
    connection_factory: ConnectionFactory,
    receive_timeout_seconds: float = 60.0,
) -> None:
    """Serve interactive streaming turns over WebSocket.

    Client messages:
    - ``{"type": "input", "input": "...", "user_id": "...", "session_id": "..."}``
    - ``{"type": "abort"}``
    - ``{"type": "ping"}``
    - ``{"type": "close"}``
    """

    if receive_timeout_seconds <= 0:
        raise ValueError("receive_timeout_seconds must be > 0")

    active: _ActiveTurn | None = None

    async def _cancel_active(reason: str) -> None:
        nonlocal active
        if not active:
            return
        active.task.cancel()
        with suppress(asyncio.CancelledError):
            await active.task

        await websocket.send_json(
            build_stream_event(
                event_type=EVENT_TURN_END,
                request_id=active.request_id,
                sequence=10_000,
                user_id=active.user_id,
                session_id=active.session_id,
                payload={"reason": reason},
            )
        )
        active = None

    async def _pump_turn(
        *, request_id: str, user_id: str | None, session_id: str | None, text: str
    ) -> None:
        try:
            async with connection_factory() as connection:
                async for event in iter_bidi_turn_events(
                    connection=connection,
                    request={"input": text, "user_id": user_id},
                    request_id=request_id,
                    user_id=user_id,
                    session_id=session_id,
                    receive_timeout_seconds=receive_timeout_seconds,
                ):
                    await websocket.send_json(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await websocket.send_json(
                build_stream_event(
                    event_type=EVENT_ERROR,
                    request_id=request_id,
                    sequence=9_999,
                    user_id=user_id,
                    session_id=session_id,
                    payload={
                        "error": f"{type(exc).__name__}: {exc}",
                        "recoverable": True,
                    },
                )
            )

    while True:
        message = await websocket.receive_json()
        message_type = str(message.get("type", "input")).strip().lower()

        if message_type == "ping":
            await websocket.send_json({"type": "pong"})
            continue

        if message_type == "abort":
            await _cancel_active(reason="aborted")
            continue

        if message_type == "close":
            await _cancel_active(reason="closed")
            close = getattr(websocket, "close", None)
            if callable(close):
                result = close()
                if isinstance(result, Awaitable):
                    await result
            break

        text = str(message.get("input", "")).strip()
        if not text:
            await websocket.send_json(
                build_stream_event(
                    event_type=EVENT_ERROR,
                    request_id=str(uuid4()),
                    sequence=1,
                    user_id=message.get("user_id"),
                    session_id=message.get("session_id"),
                    payload={"error": "input is required", "recoverable": True},
                )
            )
            continue

        await _cancel_active(reason="superseded")

        request_id = str(message.get("request_id") or uuid4())
        user_id = message.get("user_id")
        session_id = message.get("session_id")
        task = asyncio.create_task(
            _pump_turn(
                request_id=request_id,
                user_id=str(user_id) if user_id is not None else None,
                session_id=str(session_id) if session_id is not None else None,
                text=text,
            )
        )
        active = _ActiveTurn(
            request_id=request_id,
            user_id=str(user_id) if user_id is not None else None,
            session_id=str(session_id) if session_id is not None else None,
            task=task,
        )
