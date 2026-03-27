import os
import unittest
from unittest.mock import patch

from f1_agent.code_execution import _build_code_template, run_analytical_code


class CodeExecutionToolTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_disabled_by_default(self):
        result = run_analytical_code(
            "summary_stats", '{"rows": [{"points": 10}], "field": "points"}'
        )
        self.assertEqual(result["status"], "disabled")

    @patch.dict(
        os.environ,
        {
            "F1_CODE_EXECUTION_ENABLED": "true",
            "F1_RAG_PROJECT_ID": "f1-regulations-agent-chat",
            "F1_CODE_EXECUTION_LOCATION": "us-central1",
            "F1_CODE_EXECUTION_AGENT_ENGINE_NAME": "projects/p/locations/us-central1/reasoningEngines/r",
        },
        clear=True,
    )
    def test_rejects_invalid_json_payload(self):
        result = run_analytical_code("summary_stats", "{not json}")
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid JSON payload", result["message"])

    @patch.dict(
        os.environ,
        {
            "F1_CODE_EXECUTION_ENABLED": "true",
            "F1_RAG_PROJECT_ID": "f1-regulations-agent-chat",
            "F1_CODE_EXECUTION_LOCATION": "europe-west4",
            "F1_CODE_EXECUTION_AGENT_ENGINE_NAME": "projects/p/locations/us-central1/reasoningEngines/r",
        },
        clear=True,
    )
    def test_requires_us_central1(self):
        result = run_analytical_code(
            "summary_stats", '{"rows": [{"points": 10}], "field": "points"}'
        )
        self.assertEqual(result["status"], "configuration_error")
        self.assertIn("us-central1", result["message"])

    @patch.dict(
        os.environ,
        {
            "F1_CODE_EXECUTION_ENABLED": "true",
            "F1_RAG_PROJECT_ID": "f1-regulations-agent-chat",
            "F1_CODE_EXECUTION_LOCATION": "us-central1",
            "F1_CODE_EXECUTION_AGENT_ENGINE_NAME": "projects/p/locations/us-central1/reasoningEngines/r",
        },
        clear=True,
    )
    def test_rejects_unknown_task(self):
        result = run_analytical_code("unknown", "{}")
        self.assertEqual(result["status"], "error")
        self.assertIn("Unsupported task_type", result["message"])


class CodeTemplateTests(unittest.TestCase):
    def test_summary_template_embeds_field(self):
        code = _build_code_template(
            "summary_stats",
            {"rows": [{"points": 25}, {"points": 18}], "field": "points"},
        )
        self.assertIn("statistics", code)
        self.assertIn("field = payload['field']", code)

    def test_distribution_template_has_bins(self):
        code = _build_code_template(
            "distribution_bins",
            {"values": [1, 2, 3], "bins": 4},
        )
        self.assertIn("counts", code)
        self.assertIn("bins", code)


if __name__ == "__main__":
    unittest.main()
