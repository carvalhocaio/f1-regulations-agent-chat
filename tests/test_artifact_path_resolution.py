import tempfile
import unittest
from pathlib import Path

from f1_agent.db import _resolve_db_dir
from f1_agent.rag import _resolve_vector_store_dir


class ArtifactPathResolutionTests(unittest.TestCase):
    def test_resolve_vector_store_dir_prefers_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "index.faiss").touch()
            (base / "index.pkl").touch()

            resolved = _resolve_vector_store_dir(base)

            self.assertEqual(resolved, base)

    def test_resolve_vector_store_dir_supports_nested_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            nested = base / "vector_store"
            nested.mkdir(parents=True)
            (nested / "index.faiss").touch()
            (nested / "index.pkl").touch()

            resolved = _resolve_vector_store_dir(base)

            self.assertEqual(resolved, nested)

    def test_resolve_db_dir_prefers_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "f1_history.db").touch()

            resolved = _resolve_db_dir(base)

            self.assertEqual(resolved, base)

    def test_resolve_db_dir_supports_nested_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            nested = base / "f1_data"
            nested.mkdir(parents=True)
            (nested / "f1_history.db").touch()

            resolved = _resolve_db_dir(base)

            self.assertEqual(resolved, nested)


if __name__ == "__main__":
    unittest.main()
