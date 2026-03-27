"""FastAPI WebSocket server bridging clients to Agent Engine bidi stream.

Run locally:

```bash
uv run python deployment/websocket_bidi_server.py \
  --project-id "<PROJECT_ID>" \
  --location "us-central1" \
  --resource-name "projects/.../locations/us-central1/reasoningEngines/..."
```

WebSocket endpoint: ``/ws/chat``
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import uvicorn
import vertexai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai

from f1_agent.websocket_bridge import (
    build_agent_connection_factory,
    websocket_bidi_loop,
)


def create_app(
    *,
    connection_factory: Callable[[], Any],
    receive_timeout_seconds: float = 60.0,
) -> FastAPI:
    """Create FastAPI app exposing `WebSocket <-> bidi` bridge endpoint."""
    app = FastAPI(title="F1 Agent Bidi Bridge", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await websocket_bidi_loop(
                websocket=websocket,
                connection_factory=connection_factory,
                receive_timeout_seconds=receive_timeout_seconds,
            )
        except WebSocketDisconnect:
            return
        except Exception:
            with suppress(Exception):
                await websocket.close(code=1011)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FastAPI WebSocket bridge for Agent Engine bidi streaming"
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--resource-name", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--receive-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.port < 1 or args.port > 65535:
        raise ValueError("--port must be in range 1..65535")
    if args.receive_timeout_seconds <= 0:
        raise ValueError("--receive-timeout-seconds must be > 0")

    client = genai.Client(
        vertexai=True,
        project=args.project_id,
        location=args.location,
    )
    connection_factory = build_agent_connection_factory(
        client=client,
        agent_engine=args.resource_name,
        remote_agent=vertexai.Client(
            project=args.project_id,
            location=args.location,
        ).agent_engines.get(name=args.resource_name),
        class_method="bidi_stream_query",
    )

    app = create_app(
        connection_factory=connection_factory,
        receive_timeout_seconds=args.receive_timeout_seconds,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
