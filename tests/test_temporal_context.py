import unittest
from unittest.mock import patch

from f1_agent.callbacks import (
    _query_requires_web_data,
    _runtime_temporal_addendum,
    check_cache,
)


class _FakeContent:
    def __init__(self, role, texts):
        self.role = role
        self.parts = [type("P", (), {"text": t})() for t in texts]


class _FakeContext:
    def __init__(self, user_text=None):
        if user_text:
            self.user_content = _FakeContent("user", [user_text])
        else:
            self.user_content = None


class _FakeRequest:
    def __init__(self, user_text=None):
        if user_text:
            self.contents = [_FakeContent("user", [user_text])]
        else:
            self.contents = []


class _FakeCache:
    def __init__(self, answer="cached"):
        self.answer = answer
        self.get_calls = 0

    def get(self, _question):
        self.get_calls += 1
        return self.answer


class TemporalContextTests(unittest.TestCase):
    def test_query_requires_web_for_post_2024_year(self):
        self.assertTrue(_query_requires_web_data("Pódio do GP dos EUA em 2025?"))

    def test_query_requires_web_for_live_terms(self):
        self.assertTrue(_query_requires_web_data("Quem lidera o campeonato agora?"))

    def test_query_does_not_require_web_for_historical_year(self):
        self.assertFalse(_query_requires_web_data("Who won the 2023 championship?"))

    @patch("f1_agent.callbacks._current_year", return_value=2026)
    def test_runtime_addendum_contains_dynamic_year(self, _mock_current_year):
        addendum = _runtime_temporal_addendum()
        self.assertIn("Current year: 2026", addendum)
        self.assertIn("Historical DB coverage: 1950-2024 only", addendum)

    @patch("f1_agent.callbacks._get_cache")
    def test_check_cache_bypasses_time_sensitive_queries(self, mock_get_cache):
        fake_cache = _FakeCache()
        mock_get_cache.return_value = fake_cache

        ctx = _FakeContext("Pódio do GP dos EUA em 2025?")
        req = _FakeRequest()

        result = check_cache(ctx, req)
        self.assertIsNone(result)
        self.assertEqual(fake_cache.get_calls, 0)

    @patch("f1_agent.callbacks._get_cache")
    def test_check_cache_uses_cache_for_historical_queries(self, mock_get_cache):
        fake_cache = _FakeCache(answer="Verstappen")
        mock_get_cache.return_value = fake_cache

        ctx = _FakeContext("Who won the 2023 championship?")
        req = _FakeRequest()

        result = check_cache(ctx, req)
        self.assertIsNotNone(result)
        self.assertEqual(fake_cache.get_calls, 1)


if __name__ == "__main__":
    unittest.main()
