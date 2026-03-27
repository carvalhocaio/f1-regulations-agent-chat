import asyncio
import unittest
from contextlib import asynccontextmanager

from f1_agent.websocket_bridge import websocket_bidi_loop


class _FakeConnection:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def receive(self):
        await asyncio.sleep(0)
        if not self.outputs:
            raise RuntimeError("stream closed")
        return self.outputs.pop(0)


class _FakeWebSocket:
    def __init__(self, inbound):
        self.inbound = list(inbound)
        self.outbound = []
        self.closed = False

    async def receive_json(self):
        await asyncio.sleep(0.01)
        if not self.inbound:
            return {"type": "close"}
        return self.inbound.pop(0)

    async def send_json(self, payload):
        self.outbound.append(payload)

    async def close(self):
        self.closed = True


def _connection_factory(outputs):
    @asynccontextmanager
    async def _factory():
        yield _FakeConnection(outputs)

    return _factory


class WebSocketBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_bidi_loop_streams_and_closes(self):
        ws = _FakeWebSocket(
            [
                {"type": "ping"},
                {
                    "type": "input",
                    "request_id": "req-1",
                    "user_id": "u-1",
                    "session_id": "s-1",
                    "input": "hello",
                },
                {"type": "close"},
            ]
        )

        await websocket_bidi_loop(
            websocket=ws,
            connection_factory=_connection_factory(
                [
                    {"bidiStreamOutput": {"output": "Hello"}},
                    {"bidiStreamOutput": {"output": "end of turn"}},
                ]
            ),
        )

        self.assertTrue(ws.closed)
        self.assertEqual(ws.outbound[0], {"type": "pong"})
        event_types = [
            item["event_type"]
            for item in ws.outbound
            if isinstance(item, dict) and "event_type" in item
        ]
        self.assertIn("turn_start", event_types)
        self.assertIn("delta", event_types)
        self.assertIn("turn_end", event_types)

    async def test_websocket_bidi_loop_abort_emits_turn_end(self):
        ws = _FakeWebSocket(
            [
                {
                    "type": "input",
                    "request_id": "req-abort",
                    "input": "long turn",
                },
                {"type": "abort"},
                {"type": "close"},
            ]
        )

        await websocket_bidi_loop(
            websocket=ws,
            connection_factory=_connection_factory(
                [{"bidiStreamOutput": {"output": "chunk"}}]
            ),
        )

        reasons = [
            item.get("payload", {}).get("reason")
            for item in ws.outbound
            if isinstance(item, dict) and item.get("event_type") == "turn_end"
        ]
        self.assertIn("aborted", reasons)

    async def test_websocket_bidi_loop_forwards_response_contract_id(self):
        captured: list[_FakeConnection] = []

        @asynccontextmanager
        async def _factory():
            conn = _FakeConnection(
                [
                    {"bidiStreamOutput": {"output": "ok"}},
                    {"bidiStreamOutput": {"output": "end of turn"}},
                ]
            )
            captured.append(conn)
            yield conn

        ws = _FakeWebSocket(
            [
                {
                    "type": "input",
                    "request_id": "req-structured",
                    "input": "compare these",
                    "response_contract_id": "comparison_table_v1",
                },
                {"type": "close"},
            ]
        )

        await websocket_bidi_loop(websocket=ws, connection_factory=_factory)

        self.assertTrue(captured)
        sent_payload = captured[0].sent[0]
        self.assertEqual(
            sent_payload.get("response_contract_id"), "comparison_table_v1"
        )


if __name__ == "__main__":
    unittest.main()
