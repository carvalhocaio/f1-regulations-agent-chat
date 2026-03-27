import importlib.util
import json
import sys
import unittest
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "deployment"
        / "benchmark_retrieval_backends.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_retrieval_backends", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_module()


class BenchmarkRetrievalBackendsTests(unittest.TestCase):
    def test_percentile(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(bench._percentile(values, 0.5), 30.0)
        self.assertEqual(bench._percentile(values, 0.95), 50.0)

    def test_load_queries_jsonl(self):
        path = Path("/tmp/test-benchmark-queries.jsonl")
        path.write_text(
            "\n".join(
                [
                    json.dumps({"query": "What is DRS?", "expected_ids": ["a"]}),
                    json.dumps({"query": "Explain parc ferme", "expected_ids": []}),
                ]
            ),
            encoding="utf-8",
        )
        cases = bench._load_queries(path)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].query, "What is DRS?")
        self.assertEqual(cases[0].expected_ids, ["a"])
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
