"""Tests for the semantic cache module.

These tests mock the embedding model to avoid API calls and test the cache
logic in isolation (similarity computation, TTL, eviction, etc.).
"""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _fake_embed(text):
    """Deterministic fake embedding: hash text into a 256-dim unit vector."""
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(256).astype(np.float32)
    return vec / np.linalg.norm(vec)


class SemanticCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        # Patch the embedding model before importing SemanticCache
        patcher = patch(
            "f1_agent.cache._get_embeddings",
            return_value=type(
                "FakeEmbed", (), {"embed_query": staticmethod(_fake_embed)}
            )(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        from f1_agent.cache import SemanticCache

        self.cache = SemanticCache(cache_dir=Path(self._tmpdir))

    def test_put_and_get_exact_match(self):
        self.cache.put("Who won 2023?", "Verstappen won 2023.")
        result = self.cache.get("Who won 2023?")
        self.assertEqual(result, "Verstappen won 2023.")

    def test_miss_on_different_question(self):
        self.cache.put("Who won 2023?", "Verstappen won 2023.")
        result = self.cache.get("What is the fastest lap at Monza?")
        self.assertIsNone(result)

    def test_clear_removes_all(self):
        self.cache.put("Q1", "A1")
        self.cache.put("Q2", "A2")
        self.cache.clear()
        self.assertIsNone(self.cache.get("Q1"))
        self.assertIsNone(self.cache.get("Q2"))

    def test_expired_entries_return_none(self):
        # Patch TTL to very short
        self.cache.put("Q", "A", web_source=False)
        # Manually expire by updating created_at far in the past (> 30 days)
        self.cache._conn.execute(
            "UPDATE cache_entries SET created_at = ?",
            (time.time() - 90 * 24 * 3600,),
        )
        self.cache._conn.commit()
        result = self.cache.get("Q")
        self.assertIsNone(result)

    def test_hit_count_incremented(self):
        self.cache.put("Who won 2023?", "Verstappen")
        self.cache.get("Who won 2023?")
        self.cache.get("Who won 2023?")
        row = self.cache._conn.execute(
            "SELECT hit_count FROM cache_entries LIMIT 1"
        ).fetchone()
        self.assertEqual(row[0], 2)

    def test_web_source_flag_stored(self):
        self.cache.put("Q", "A", web_source=True)
        row = self.cache._conn.execute(
            "SELECT web_source FROM cache_entries LIMIT 1"
        ).fetchone()
        self.assertEqual(row[0], 1)

    def test_lookup_returns_metadata(self):
        self.cache.put("Who won 2023?", "Verstappen")
        result = self.cache.lookup("Who won 2023?")
        self.assertEqual(result.outcome, "hit")
        self.assertEqual(result.answer, "Verstappen")
        self.assertGreaterEqual(result.candidates_scanned, 1)
        self.assertIsNotNone(result.similarity_top1)

    def test_sweep_removes_expired_entries(self):
        from f1_agent.cache import SemanticCache

        cache = SemanticCache(
            cache_dir=Path(self._tmpdir),
            sweep_every_ops=1,
            sweep_interval_seconds=1,
        )
        cache.put("Q", "A", web_source=False)
        cache._conn.execute(
            "UPDATE cache_entries SET created_at = ?",
            (time.time() - 90 * 24 * 3600,),
        )
        cache._conn.commit()

        result = cache.lookup("Q")
        self.assertIsNone(result.answer)
        rows = cache._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        self.assertEqual(rows[0], 0)

    def test_max_entries_prunes_lowest_priority(self):
        from f1_agent.cache import SemanticCache

        cache = SemanticCache(
            cache_dir=Path(self._tmpdir),
            max_entries=2,
            sweep_every_ops=1,
            sweep_interval_seconds=1,
        )
        cache.put("Q1", "A1")
        cache.put("Q2", "A2")
        cache.put("Q3", "A3")

        rows = cache._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        self.assertEqual(rows[0], 2)


if __name__ == "__main__":
    unittest.main()
