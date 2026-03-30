"""Tests for get_current_season_info tool."""

import json
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from f1_agent.tools import get_current_season_info


def _fake_races(dates: list[str], season: str = "2026") -> list[dict]:
    """Build minimal race entries for testing."""
    races = []
    for i, d in enumerate(dates, 1):
        races.append(
            {
                "season": season,
                "round": str(i),
                "raceName": f"Race {i}",
                "date": d,
                "Circuit": {
                    "circuitName": f"Circuit {i}",
                    "Location": {"country": f"Country {i}"},
                },
            }
        )
    return races


class TestGetCurrentSeasonInfo(unittest.TestCase):
    def setUp(self):
        # Reset cache between tests
        from f1_agent import tools_jolpica

        tools_jolpica._season_cache["data"] = None
        tools_jolpica._season_cache["fetched_at"] = 0.0

    @patch("f1_agent.tools_jolpica._fetch_season_calendar")
    @patch("f1_agent.tools_jolpica.datetime")
    def test_returns_completed_and_upcoming(self, mock_dt, mock_fetch):
        mock_dt.now.return_value.date.return_value = date(2026, 3, 29)
        mock_dt.side_effect = lambda *a, **kw: date(*a, **kw) if a else mock_dt
        mock_fetch.return_value = _fake_races(
            ["2026-03-08", "2026-03-15", "2026-03-29", "2026-04-12", "2026-04-26"]
        )

        result = get_current_season_info()

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["season_started"])
        self.assertEqual(result["completed_count"], 3)
        self.assertEqual(result["upcoming_count"], 2)
        self.assertEqual(result["total_races"], 5)
        self.assertIsNotNone(result["next_race"])
        self.assertEqual(result["next_race"]["date"], "2026-04-12")

    @patch("f1_agent.tools_jolpica._fetch_season_calendar")
    @patch("f1_agent.tools_jolpica.datetime")
    def test_pre_season(self, mock_dt, mock_fetch):
        mock_dt.now.return_value.date.return_value = date(2026, 1, 15)
        mock_dt.side_effect = lambda *a, **kw: date(*a, **kw) if a else mock_dt
        mock_fetch.return_value = _fake_races(["2026-03-08", "2026-03-15"])

        result = get_current_season_info()

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["season_started"])
        self.assertEqual(result["completed_count"], 0)

    @patch("f1_agent.tools_jolpica._fetch_season_calendar")
    def test_api_unavailable(self, mock_fetch):
        mock_fetch.return_value = []

        result = get_current_season_info()

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("google_search", result["message"])

    @patch("f1_agent.tools_jolpica._fetch_season_calendar")
    @patch("f1_agent.tools_jolpica.datetime")
    def test_all_races_completed(self, mock_dt, mock_fetch):
        mock_dt.now.return_value.date.return_value = date(2026, 12, 15)
        mock_dt.side_effect = lambda *a, **kw: date(*a, **kw) if a else mock_dt
        mock_fetch.return_value = _fake_races(["2026-03-08", "2026-11-29"])

        result = get_current_season_info()

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["season_started"])
        self.assertEqual(result["completed_count"], 2)
        self.assertEqual(result["upcoming_count"], 0)
        self.assertIsNone(result["next_race"])


if __name__ == "__main__":
    unittest.main()
