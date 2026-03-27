import unittest

from f1_agent.streaming_protocol import (
    EVENT_DELTA,
    EVENT_TURN_END,
    build_stream_event,
    extract_bidi_text,
    is_bidi_end_of_turn,
)


class StreamingProtocolTests(unittest.TestCase):
    def test_build_stream_event_includes_protocol_fields(self):
        event = build_stream_event(
            event_type=EVENT_DELTA,
            request_id="req-1",
            sequence=2,
            user_id="u1",
            session_id="s1",
            payload={"text": "hello"},
        )

        self.assertEqual(event["stream_protocol_version"], "v1")
        self.assertEqual(event["event_type"], EVENT_DELTA)
        self.assertEqual(event["request_id"], "req-1")
        self.assertEqual(event["sequence"], 2)
        self.assertEqual(event["payload"], {"text": "hello"})
        self.assertIn("ts", event)

    def test_end_of_turn_detection_matches_doc_shape(self):
        raw = {"bidiStreamOutput": {"output": "end of turn"}}
        self.assertTrue(is_bidi_end_of_turn(raw))

        not_end = {"bidiStreamOutput": {"output": "partial text"}}
        self.assertFalse(is_bidi_end_of_turn(not_end))

    def test_extract_bidi_text_filters_end_of_turn(self):
        raw_delta = {"bidiStreamOutput": {"output": "hello"}}
        self.assertEqual(extract_bidi_text(raw_delta), "hello")

        raw_end = {"bidiStreamOutput": {"output": "end of turn"}}
        self.assertIsNone(extract_bidi_text(raw_end))

    def test_invalid_event_type_raises(self):
        with self.assertRaises(ValueError):
            build_stream_event(
                event_type=EVENT_TURN_END + "_invalid",
                request_id="req-1",
                sequence=1,
                user_id=None,
                session_id=None,
            )


if __name__ == "__main__":
    unittest.main()
