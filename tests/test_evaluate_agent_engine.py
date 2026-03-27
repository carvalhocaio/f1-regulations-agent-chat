import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "deployment" / "evaluate_agent_engine.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_agent_engine", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate_agent_engine = _load_module()


class EvaluateAgentEngineTests(unittest.TestCase):
    def test_load_cases_parses_required_fields(self):
        rows = [
            {
                "id": "case-1",
                "category": "tool_routing",
                "criticality": "high",
                "prompt": "Test prompt",
                "reference": "Expected",
            },
            {
                "id": "case-2",
                "category": "factuality",
                "criticality": "medium",
                "prompt": "Another prompt",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            cases = evaluate_agent_engine._load_cases(path)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].case_id, "case-1")
        self.assertEqual(cases[0].reference, "Expected")
        self.assertEqual(cases[1].reference, None)

    def test_extract_summary_scores_handles_nested_payloads(self):
        summary = {
            "final_response_quality": {"score": 4.2},
            "tool_use_quality": {"value": 4.0},
            "nested": {
                "hallucination": {"mean": 4.4},
                "safety": {"avg": 4.6},
            },
        }
        metrics = [
            "FINAL_RESPONSE_QUALITY",
            "TOOL_USE_QUALITY",
            "HALLUCINATION",
            "SAFETY",
        ]

        scores = evaluate_agent_engine._extract_summary_scores(summary, metrics)

        self.assertEqual(scores["FINAL_RESPONSE_QUALITY"], 4.2)
        self.assertEqual(scores["TOOL_USE_QUALITY"], 4.0)
        self.assertEqual(scores["HALLUCINATION"], 4.4)
        self.assertEqual(scores["SAFETY"], 4.6)


if __name__ == "__main__":
    unittest.main()
