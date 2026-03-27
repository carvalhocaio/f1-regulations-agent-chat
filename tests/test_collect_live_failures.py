import importlib.util
import unittest
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "deployment" / "collect_live_failures.py"
    )
    spec = importlib.util.spec_from_file_location("collect_live_failures", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_live_failures = _load_module()


class CollectLiveFailuresTests(unittest.TestCase):
    def test_collect_from_eval_artifacts_filters_by_failed_metric(self):
        eval_rows = [
            {
                "id": "tool-001",
                "category": "tool_routing",
                "criticality": "high",
                "prompt": "Use a ferramenta search para achar regra X.",
            },
            {
                "id": "hist-001",
                "category": "historical_fact",
                "criticality": "medium",
                "prompt": "Who won 2012 Brazilian GP?",
            },
        ]
        gate = {
            "checks": [
                {"metric": "TOOL_USE_QUALITY", "passed": False},
                {"metric": "FINAL_RESPONSE_QUALITY", "passed": True},
            ]
        }
        report = {"summary_metrics": {"TOOL_USE_QUALITY": 3.6}}

        failures = collect_live_failures.collect_from_eval_artifacts(
            eval_dataset_rows=eval_rows,
            eval_gate_result=gate,
            eval_report=report,
        )

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["category"], "tool_routing")
        self.assertEqual(failures[0]["source"], "eval_pipeline")

    def test_collect_from_runtime_events_parses_tool_validation(self):
        events = [
            {
                "message": "tool_validation_error | tool=query_f1_history code=INVALID_ARGUMENT count=2",
                "prompt": "Quantas vitorias o Senna teve?",
            }
        ]

        failures = collect_live_failures.collect_from_runtime_events(events)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["failure_type"], "tool_validation_error")
        self.assertEqual(failures[0]["category"], "tool_routing")
        self.assertEqual(failures[0]["source"], "runtime_logs")

    def test_collect_from_runtime_events_parses_structured_validation(self):
        events = [
            {
                "log": "structured_response_validation | contract_id=sources_block_v1 outcome=schema_failure error=missing_field"
            }
        ]
        failures = collect_live_failures.collect_from_runtime_events(events)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["failure_type"], "structured_response_validation")
        self.assertEqual(failures[0]["category"], "response_format")


if __name__ == "__main__":
    unittest.main()
