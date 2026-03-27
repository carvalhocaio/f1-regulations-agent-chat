import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "deployment" / "eval_gate.py"
    spec = importlib.util.spec_from_file_location("eval_gate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_gate = _load_module()


class EvalGateTests(unittest.TestCase):
    def test_passes_with_absolute_and_regression_checks(self):
        report = {
            "summary_metrics": {
                "FINAL_RESPONSE_QUALITY": 4.2,
                "TOOL_USE_QUALITY": 4.0,
            }
        }
        baseline = {
            "summary_metrics": {
                "FINAL_RESPONSE_QUALITY": 4.3,
                "TOOL_USE_QUALITY": 4.05,
            }
        }
        thresholds = {
            "metrics": {
                "FINAL_RESPONSE_QUALITY": {
                    "absolute_min": 4.0,
                    "max_regression_delta": 0.2,
                },
                "TOOL_USE_QUALITY": {
                    "absolute_min": 3.8,
                    "max_regression_delta": 0.2,
                },
            }
        }

        result = eval_gate.evaluate_gate(
            report=report,
            thresholds=thresholds,
            baseline=baseline,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_fails_when_absolute_min_not_met(self):
        report = {
            "summary_metrics": {
                "FINAL_RESPONSE_QUALITY": 3.6,
            }
        }
        thresholds = {
            "metrics": {
                "FINAL_RESPONSE_QUALITY": {
                    "absolute_min": 3.8,
                    "max_regression_delta": 0.2,
                }
            }
        }

        result = eval_gate.evaluate_gate(
            report=report,
            thresholds=thresholds,
            baseline=None,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("absolute_min", result["failures"][0])

    def test_fails_when_regression_exceeds_delta(self):
        report = {
            "summary_metrics": {
                "TOOL_USE_QUALITY": 3.7,
            }
        }
        baseline = {
            "summary_metrics": {
                "TOOL_USE_QUALITY": 4.1,
            }
        }
        thresholds = {
            "metrics": {
                "TOOL_USE_QUALITY": {
                    "absolute_min": 3.6,
                    "max_regression_delta": 0.2,
                }
            }
        }

        result = eval_gate.evaluate_gate(
            report=report,
            thresholds=thresholds,
            baseline=baseline,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("max_regression_delta", result["failures"][0])


if __name__ == "__main__":
    unittest.main()
