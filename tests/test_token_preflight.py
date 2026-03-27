"""Tests for the token preflight check module."""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from f1_agent.token_preflight import (
    PreflightResult,
    _identify_injected_blocks,
    check_and_truncate,
)


# ---------------------------------------------------------------------------
# Fakes (same pattern as test_model_routing.py)
# ---------------------------------------------------------------------------


class _FakeContent:
    def __init__(self, role, text):
        self.role = role
        self.parts = [type("P", (), {"text": text})()]


class _FakeRequest:
    def __init__(self, model="gemini-2.5-pro", contents=None):
        self.model = model
        self.contents = contents or []
        self.config = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request_with_injected_blocks():
    """Build a request with all 4 injected blocks + real conversation."""
    contents = [
        _FakeContent(
            "user",
            "\n\n## Dynamic few-shot examples from real corrected errors\n"
            "- example 1\n- example 2",
        ),
        _FakeContent(
            "user",
            "\n\n## Long-term user memory (cross-session)\n"
            "Use only if relevant.\n- fact 1",
        ),
        _FakeContent(
            "user",
            "## User corrections from this session\n"
            "- correction 1",
        ),
        _FakeContent(
            "user",
            "\n\n## Runtime temporal context — OVERRIDES YOUR TRAINING DATA\n"
            "- Today (UTC): 2026-03-26",
        ),
        _FakeContent("user", "Who won the 2023 championship?"),
        _FakeContent("model", "Max Verstappen won in 2023."),
    ]
    return _FakeRequest(contents=contents)


# ---------------------------------------------------------------------------
# Tests: block identification
# ---------------------------------------------------------------------------


class TestIdentifyInjectedBlocks(unittest.TestCase):
    def test_identifies_all_blocks(self):
        req = _make_request_with_injected_blocks()
        blocks = _identify_injected_blocks(req.contents)
        categories = [cat for _, cat in blocks]
        self.assertEqual(
            categories, ["examples", "memories", "corrections", "temporal"]
        )

    def test_no_injected_blocks(self):
        req = _FakeRequest(contents=[
            _FakeContent("user", "Who won in 2023?"),
            _FakeContent("model", "Verstappen."),
        ])
        blocks = _identify_injected_blocks(req.contents)
        self.assertEqual(blocks, [])

    def test_partial_blocks(self):
        contents = [
            _FakeContent(
                "user",
                "## User corrections from this session\n- fix 1",
            ),
            _FakeContent(
                "user",
                "\n\n## Runtime temporal context — OVERRIDES YOUR TRAINING DATA\n"
                "- Today: 2026-03-26",
            ),
            _FakeContent("user", "Hello"),
        ]
        blocks = _identify_injected_blocks(contents)
        categories = [cat for _, cat in blocks]
        self.assertEqual(categories, ["corrections", "temporal"])

    def test_ignores_model_role(self):
        contents = [
            _FakeContent(
                "model",
                "## Dynamic few-shot examples from real corrected errors\nfake",
            ),
        ]
        blocks = _identify_injected_blocks(contents)
        self.assertEqual(blocks, [])


# ---------------------------------------------------------------------------
# Tests: disabled by default
# ---------------------------------------------------------------------------


class TestDisabledByDefault(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=False)
    def test_returns_none_when_disabled(self):
        req = _make_request_with_injected_blocks()
        result = check_and_truncate(req)
        self.assertIsNone(result)

    @patch.dict(
        "os.environ",
        {"F1_PREFLIGHT_TOKEN_CHECK_ENABLED": "false"},
        clear=False,
    )
    def test_returns_none_when_explicitly_disabled(self):
        req = _make_request_with_injected_blocks()
        result = check_and_truncate(req)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tests: under threshold — no truncation
# ---------------------------------------------------------------------------


class TestUnderThreshold(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {"F1_PREFLIGHT_TOKEN_CHECK_ENABLED": "true"},
        clear=False,
    )
    @patch("f1_agent.token_preflight.count_request_tokens", return_value=5000)
    def test_no_truncation_when_under(self, mock_count):
        req = _make_request_with_injected_blocks()
        original_len = len(req.contents)

        result = check_and_truncate(req)

        self.assertIsInstance(result, PreflightResult)
        self.assertEqual(result.original_tokens, 5000)
        self.assertEqual(result.final_tokens, 5000)
        self.assertEqual(result.removed, [])
        self.assertFalse(result.truncated)
        self.assertEqual(len(req.contents), original_len)
        mock_count.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: over threshold — progressive truncation
# ---------------------------------------------------------------------------


class TestProgressiveTruncation(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "F1_PREFLIGHT_TOKEN_CHECK_ENABLED": "true",
            "F1_PREFLIGHT_TOKEN_HARD_LIMIT": "10000",
        },
        clear=False,
    )
    @patch("f1_agent.token_preflight.count_request_tokens")
    def test_removes_examples_first(self, mock_count):
        # First call: over threshold; second call (after removing examples): under
        mock_count.side_effect = [15000, 8000]
        req = _make_request_with_injected_blocks()

        result = check_and_truncate(req)

        self.assertIsInstance(result, PreflightResult)
        self.assertEqual(result.removed, ["examples"])
        self.assertTrue(result.truncated)
        self.assertEqual(result.final_tokens, 8000)
        # Verify examples block was removed
        texts = [p.parts[0].text for p in req.contents if p.role == "user"]
        self.assertFalse(
            any("Dynamic few-shot examples" in t for t in texts)
        )

    @patch.dict(
        "os.environ",
        {
            "F1_PREFLIGHT_TOKEN_CHECK_ENABLED": "true",
            "F1_PREFLIGHT_TOKEN_HARD_LIMIT": "10000",
        },
        clear=False,
    )
    @patch("f1_agent.token_preflight.count_request_tokens")
    def test_truncation_order(self, mock_count):
        # Over on every call until all blocks removed
        mock_count.side_effect = [50000, 40000, 30000, 20000, 8000]
        req = _make_request_with_injected_blocks()

        result = check_and_truncate(req)

        self.assertEqual(
            result.removed,
            ["examples", "memories", "corrections", "temporal"],
        )
        self.assertTrue(result.truncated)
        # Only real conversation should remain
        self.assertEqual(len(req.contents), 2)

    @patch.dict(
        "os.environ",
        {
            "F1_PREFLIGHT_TOKEN_CHECK_ENABLED": "true",
            "F1_PREFLIGHT_TOKEN_HARD_LIMIT": "10000",
        },
        clear=False,
    )
    @patch("f1_agent.token_preflight.count_request_tokens")
    def test_still_over_after_all_removed(self, mock_count):
        # Still over threshold even after removing all blocks
        mock_count.return_value = 50000
        req = _make_request_with_injected_blocks()

        result = check_and_truncate(req)

        self.assertEqual(
            result.removed,
            ["examples", "memories", "corrections", "temporal"],
        )
        self.assertEqual(result.final_tokens, 50000)
        # Real conversation preserved
        self.assertEqual(len(req.contents), 2)


# ---------------------------------------------------------------------------
# Tests: no injected blocks to remove
# ---------------------------------------------------------------------------


class TestNoBlocksToRemove(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "F1_PREFLIGHT_TOKEN_CHECK_ENABLED": "true",
            "F1_PREFLIGHT_TOKEN_HARD_LIMIT": "100",
        },
        clear=False,
    )
    @patch("f1_agent.token_preflight.count_request_tokens", return_value=5000)
    def test_conversation_not_touched(self, mock_count):
        req = _FakeRequest(contents=[
            _FakeContent("user", "Who won in 2023?"),
            _FakeContent("model", "Verstappen."),
        ])

        result = check_and_truncate(req)

        self.assertEqual(result.removed, [])
        # Contents not modified
        self.assertEqual(len(req.contents), 2)


# ---------------------------------------------------------------------------
# Tests: API failure — graceful degradation
# ---------------------------------------------------------------------------


class TestApiFailure(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {"F1_PREFLIGHT_TOKEN_CHECK_ENABLED": "true"},
        clear=False,
    )
    @patch(
        "f1_agent.token_preflight.count_request_tokens",
        side_effect=RuntimeError("API unavailable"),
    )
    def test_exception_propagates_to_callback(self, mock_count):
        """The module raises; the callback wrapper in callbacks.py catches it."""
        from f1_agent.callbacks import preflight_token_check

        req = _make_request_with_injected_blocks()
        original_len = len(req.contents)
        ctx = MagicMock()

        # Should not raise — callback catches exceptions
        result = preflight_token_check(ctx, req)

        self.assertIsNone(result)
        # Contents untouched
        self.assertEqual(len(req.contents), original_len)


# ---------------------------------------------------------------------------
# Tests: threshold configuration
# ---------------------------------------------------------------------------


class TestThresholdConfig(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "F1_PREFLIGHT_TOKEN_CHECK_ENABLED": "true",
            "F1_PREFLIGHT_TOKEN_THRESHOLD": "0.50",
        },
        clear=False,
    )
    @patch("f1_agent.token_preflight.count_request_tokens")
    def test_custom_threshold_fraction(self, mock_count):
        # 50% of 1M = 524288.  600000 is over.
        mock_count.side_effect = [600000, 400000]
        req = _make_request_with_injected_blocks()

        result = check_and_truncate(req)

        self.assertEqual(result.threshold, 524288)
        self.assertTrue(result.truncated)

    @patch.dict(
        "os.environ",
        {
            "F1_PREFLIGHT_TOKEN_CHECK_ENABLED": "true",
            "F1_PREFLIGHT_TOKEN_HARD_LIMIT": "5000",
        },
        clear=False,
    )
    @patch("f1_agent.token_preflight.count_request_tokens")
    def test_hard_limit_overrides_fraction(self, mock_count):
        mock_count.side_effect = [6000, 4000]
        req = _make_request_with_injected_blocks()

        result = check_and_truncate(req)

        self.assertEqual(result.threshold, 5000)
        self.assertTrue(result.truncated)


if __name__ == "__main__":
    unittest.main()
