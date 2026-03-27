import unittest
from datetime import date
from unittest.mock import patch

from f1_agent.callbacks import (
    _query_requires_web_data,
    _resolve_temporal_references,
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


class _FakeCacheLookupResult:
    def __init__(self, answer):
        self.answer = answer
        self.outcome = "hit" if answer is not None else "miss"
        self.lookup_ms = 0.1
        self.candidates_scanned = 1
        self.similarity_top1 = 0.95 if answer is not None else None
        self.evicted_count = 0


class _FakeCache:
    def __init__(self, answer="cached"):
        self.answer = answer
        self.get_calls = 0

    def get(self, _question):
        self.get_calls += 1
        return self.answer

    def lookup(self, question):
        self.get_calls += 1
        return _FakeCacheLookupResult(self.answer)


class TemporalContextTests(unittest.TestCase):
    def test_query_requires_web_for_post_2024_year(self):
        self.assertTrue(_query_requires_web_data("Pódio do GP dos EUA em 2025?"))

    def test_query_requires_web_for_live_terms(self):
        self.assertTrue(_query_requires_web_data("Quem lidera o campeonato agora?"))

    def test_query_does_not_require_web_for_historical_year(self):
        self.assertFalse(_query_requires_web_data("Who won the 2023 championship?"))

    def test_query_requires_web_for_relative_temporal_pt(self):
        self.assertTrue(
            _query_requires_web_data("quem foi o campeão da última temporada?")
        )

    def test_query_requires_web_for_relative_temporal_en(self):
        self.assertTrue(_query_requires_web_data("who was the last season champion?"))

    def test_query_requires_web_for_current_champion(self):
        self.assertTrue(_query_requires_web_data("quem é o atual campeão?"))

    def test_query_requires_web_for_last_event_without_year(self):
        self.assertTrue(
            _query_requires_web_data("Qual foi o último pódio do GP de São Paulo?")
        )

    def test_query_requires_web_for_next_gp_without_year(self):
        self.assertTrue(_query_requires_web_data("Qual é o próximo GP?"))

    @patch("f1_agent.callbacks._current_year", return_value=2026)
    def test_runtime_addendum_contains_dynamic_year(self, _mock_current_year):
        addendum = _runtime_temporal_addendum()
        self.assertIn("Current year: 2026", addendum)
        self.assertIn("Historical DB coverage: 1950-2024 only", addendum)

    @patch("f1_agent.callbacks._current_year", return_value=2026)
    def test_runtime_addendum_declares_last_season_completed(self, _mock):
        addendum = _runtime_temporal_addendum()
        self.assertIn("2025 F1 season is FULLY COMPLETED", addendum)
        self.assertIn("OVERRIDES YOUR TRAINING DATA", addendum)

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


class TemporalResolutionTests(unittest.TestCase):
    @patch("f1_agent.callbacks._current_year", return_value=2026)
    def test_resolves_last_season_pt(self, _mock):
        result = _resolve_temporal_references("quem foi o campeão da última temporada?")
        self.assertIsNotNone(result)
        self.assertIn("2025", result)
        self.assertIn("COMPLETED", result)
        self.assertIn("local database", result.lower())

    @patch("f1_agent.callbacks._current_year", return_value=2026)
    def test_resolves_last_season_en(self, _mock):
        result = _resolve_temporal_references("who won last season?")
        self.assertIsNotNone(result)
        self.assertIn("2025", result)

    @patch("f1_agent.callbacks._current_year", return_value=2026)
    def test_resolves_last_n_seasons(self, _mock):
        result = _resolve_temporal_references("últimos 3 campeões de pilotos")
        self.assertIsNotNone(result)
        self.assertIn("2023", result)
        self.assertIn("2025", result)
        # Should suggest both DB and out-of-coverage years
        self.assertIn("DB", result)
        self.assertIn("outside", result.lower())

    @patch("f1_agent.callbacks._current_year", return_value=2026)
    def test_resolves_current_champion(self, _mock):
        result = _resolve_temporal_references("quem é o atual campeão?")
        self.assertIsNotNone(result)
        self.assertIn("2025", result)
        self.assertIn("champion", result.lower())

    @patch("f1_agent.callbacks._current_year", return_value=2026)
    def test_resolves_this_season(self, _mock):
        result = _resolve_temporal_references("como está esta temporada?")
        self.assertIsNotNone(result)
        self.assertIn("2026", result)

    @patch("f1_agent.callbacks._current_date", return_value=date(2026, 6, 1))
    @patch("f1_agent.callbacks._current_year", return_value=2026)
    def test_resolves_last_event_without_year_as_completed_edition(
        self, _mock_year, _mock_date
    ):
        result = _resolve_temporal_references(
            "Qual foi o último pódio do GP de São Paulo?"
        )
        self.assertIsNotNone(result)
        self.assertIn("LAST COMPLETED edition", result)
        self.assertIn("event DATE and YEAR", result)

    @patch("f1_agent.callbacks._current_date", return_value=date(2027, 1, 15))
    @patch("f1_agent.callbacks._current_year", return_value=2027)
    def test_preseason_guard_for_current_standings(self, _mock_year, _mock_date):
        result = _resolve_temporal_references("Quem é o líder do campeonato?")
        self.assertIsNotNone(result)
        self.assertIn("Pre-season guard", result)
        self.assertIn("season has not started", result)

    def test_returns_none_for_no_relative_terms(self):
        result = _resolve_temporal_references("Quem venceu o GP de 2023?")
        self.assertIsNone(result)

    def test_returns_none_for_empty_text(self):
        result = _resolve_temporal_references("")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
