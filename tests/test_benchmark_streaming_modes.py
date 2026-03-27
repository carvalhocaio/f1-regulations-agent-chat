import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "deployment"
        / "benchmark_streaming_modes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_streaming_modes", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_module()


class BenchmarkStreamingModesTests(unittest.TestCase):
    def test_percentile_returns_expected_value(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(bench._percentile(values, 0.50), 3.0)
        self.assertEqual(bench._percentile(values, 0.95), 5.0)

    def test_summarize_reports_ttft_and_turn_metrics(self):
        results = [
            bench.ModeResult(ok=True, ttft_seconds=0.5, turn_seconds=2.0),
            bench.ModeResult(ok=True, ttft_seconds=0.8, turn_seconds=2.5),
            bench.ModeResult(
                ok=False,
                ttft_seconds=0.0,
                turn_seconds=1.0,
                error="RuntimeError: boom",
            ),
        ]

        summary = bench._summarize("bidi", results)
        self.assertEqual(summary["mode"], "bidi")
        self.assertEqual(summary["total_requests"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["error_count"], 1)
        self.assertIn("ttft_p95_seconds", summary)
        self.assertIn("turn_p95_seconds", summary)
        self.assertEqual(summary["errors"], ["RuntimeError: boom"])


if __name__ == "__main__":
    unittest.main()
