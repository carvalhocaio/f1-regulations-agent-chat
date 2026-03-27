import asyncio
import unittest
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from deployment.websocket_bidi_server import create_app


class _FakeConnection:
    def __init__(self):
        self._outputs = [
            {"bidiStreamOutput": {"output": "Hello"}},
            {"bidiStreamOutput": {"output": "end of turn"}},
        ]
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def receive(self):
        await asyncio.sleep(0)
        if not self._outputs:
            raise RuntimeError("stream closed")
        return self._outputs.pop(0)


def _factory():
    @asynccontextmanager
    async def _ctx():
        yield _FakeConnection()

    return _ctx


class WebSocketBidiServerTests(unittest.TestCase):
    def test_ws_chat_ping_and_input(self):
        app = create_app(connection_factory=_factory(), receive_timeout_seconds=5.0)
        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "ping"})
            self.assertEqual(ws.receive_json(), {"type": "pong"})

            ws.send_json(
                {
                    "type": "input",
                    "request_id": "req-1",
                    "user_id": "u1",
                    "session_id": "s1",
                    "input": "hello",
                }
            )
            event_types = [ws.receive_json()["event_type"] for _ in range(3)]
            self.assertEqual(event_types, ["turn_start", "delta", "turn_end"])


if __name__ == "__main__":
    unittest.main()
