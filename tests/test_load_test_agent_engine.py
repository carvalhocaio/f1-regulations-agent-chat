import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "deployment" / "load_test_agent_engine.py"
    )
    spec = importlib.util.spec_from_file_location("load_test_agent_engine", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


load_test = _load_module()


class LoadTestErrorMetricsTests(unittest.TestCase):
    def test_summarize_error_metrics_groups_by_status_and_type(self):
        errors = [
            load_test.RequestResult(
                ok=False,
                latency_seconds=0.1,
                error="Exception: 429 Too Many Requests",
                error_type="Exception",
                status_code=429,
                is_transient=True,
            ),
            load_test.RequestResult(
                ok=False,
                latency_seconds=0.2,
                error="RuntimeError: 503 Service Unavailable",
                error_type="RuntimeError",
                status_code=503,
                is_transient=True,
            ),
            load_test.RequestResult(
                ok=False,
                latency_seconds=0.3,
                error="ValueError: invalid input",
                error_type="ValueError",
                status_code=None,
                is_transient=False,
            ),
        ]

        summary = load_test._summarize_error_metrics(errors=errors, total_requests=10)

        self.assertEqual(summary["error_buckets"]["429:Exception"], 1)
        self.assertEqual(summary["error_buckets"]["503:RuntimeError"], 1)
        self.assertEqual(summary["error_buckets"]["unknown:ValueError"], 1)
        self.assertEqual(summary["transient_error_count"], 2)
        self.assertEqual(summary["quota_transient_error_count"], 2)
        self.assertEqual(summary["transient_error_rate"], 0.2)
        self.assertEqual(summary["quota_transient_error_rate"], 0.2)


if __name__ == "__main__":
    unittest.main()
