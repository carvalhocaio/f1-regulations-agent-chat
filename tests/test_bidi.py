import asyncio
import unittest

from f1_agent.bidi import iter_bidi_turn_events


class _FakeConnection:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def receive(self):
        await asyncio.sleep(0)
        if not self._responses:
            raise RuntimeError("no more responses")
        return self._responses.pop(0)


class BidiHelpersTests(unittest.IsolatedAsyncioTestCase):
    async def test_iter_bidi_turn_events_streams_delta_and_end(self):
        conn = _FakeConnection(
            [
                {"bidiStreamOutput": {"output": "Hello"}},
                {"bidiStreamOutput": {"output": "end of turn"}},
            ]
        )

        events = []
        async for event in iter_bidi_turn_events(
            connection=conn,
            request={"input": "hi"},
            request_id="req-123",
            user_id="user-1",
            session_id="session-1",
            receive_timeout_seconds=5,
        ):
            events.append(event)

        self.assertEqual(conn.sent, [{"input": "hi"}])
        self.assertEqual(events[0]["event_type"], "turn_start")
        self.assertEqual(events[1]["event_type"], "delta")
        self.assertEqual(events[1]["payload"]["text"], "Hello")
        self.assertEqual(events[2]["event_type"], "turn_end")

    async def test_iter_bidi_turn_events_emits_error_on_receive_failure(self):
        conn = _FakeConnection([])

        events = []
        async for event in iter_bidi_turn_events(
            connection=conn,
            request={"input": "hi"},
            request_id="req-err",
            user_id="user-1",
            session_id=None,
            receive_timeout_seconds=0.2,
        ):
            events.append(event)

        self.assertEqual(events[0]["event_type"], "turn_start")
        self.assertEqual(events[1]["event_type"], "error")
        self.assertTrue(events[1]["payload"]["recoverable"])


if __name__ == "__main__":
    unittest.main()
