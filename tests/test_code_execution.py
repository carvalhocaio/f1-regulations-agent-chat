import os
import unittest
from unittest.mock import patch

import f1_agent.code_execution as code_execution
from f1_agent.code_execution import _build_code_template, run_analytical_code


class _FakeChunk:
    def __init__(self, text: str, file_name: str | None = None):
        self.data = text.encode("utf-8")
        self.mime_type = "text/plain"
        if file_name is None:
            self.metadata = None
        else:
            self.metadata = type(
                "Metadata",
                (),
                {"attributes": {"file_name": file_name.encode("utf-8")}},
            )()


class _FakeResponse:
    def __init__(self, outputs):
        self.outputs = outputs


class _FakeOperationResponse:
    def __init__(self, name: str):
        self.name = name


class _FakeOperation:
    def __init__(self, name: str):
        self.response = _FakeOperationResponse(name)


class _FakeSandboxes:
    def __init__(self, execute_response):
        self._execute_response = execute_response
        self.create_config = None
        self.deleted_names: list[str] = []

    def create(self, *, name, spec, config):
        del spec
        self.create_config = config
        return _FakeOperation(f"{name}/sandboxEnvironments/sb-1")

    def execute_code(self, *, name, input_data):
        del name, input_data
        return self._execute_response

    def delete(self, *, name):
        self.deleted_names.append(name)


class _FakeAgentEngines:
    def __init__(self, sandboxes):
        self.sandboxes = sandboxes


class _FakeClient:
    def __init__(self, sandboxes):
        self.agent_engines = _FakeAgentEngines(sandboxes)


class CodeExecutionToolTests(unittest.TestCase):
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
    def test_executes_sandbox_and_parses_chunk_outputs(self):
        fake_response = _FakeResponse(
            [
                _FakeChunk(
                    '{"status":"success","count":2,"mean":15.0}',
                    file_name="stdout.txt",
                ),
            ]
        )
        fake_sandboxes = _FakeSandboxes(fake_response)
        fake_client = _FakeClient(fake_sandboxes)

        with (
            patch(
                "f1_agent.code_execution.vertexai.Client",
                return_value=fake_client,
            ),
            patch(
                "f1_agent.code_execution.run_with_retry",
                side_effect=lambda _name, fn, logger_instance=None: fn(),
            ),
        ):
            result = run_analytical_code(
                "summary_stats",
                '{"rows": [{"points": 10}, {"points": 20}], "field": "points"}',
            )

        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["result"])
        self.assertEqual(result["result"]["status"], "success")
        self.assertEqual(result["result"]["mean"], 15.0)
        self.assertIsInstance(
            fake_sandboxes.create_config,
            code_execution.vertexai.types.CreateAgentEngineSandboxConfig,
        )
        self.assertEqual(len(fake_sandboxes.deleted_names), 1)

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
    def test_marks_partial_success_when_stderr_chunk_is_present(self):
        fake_response = _FakeResponse(
            [
                _FakeChunk('{"status":"success","count":1}', file_name="stdout.log"),
                _FakeChunk("warning: division by zero", file_name="stderr.log"),
            ]
        )
        fake_sandboxes = _FakeSandboxes(fake_response)
        fake_client = _FakeClient(fake_sandboxes)

        with (
            patch(
                "f1_agent.code_execution.vertexai.Client",
                return_value=fake_client,
            ),
            patch(
                "f1_agent.code_execution.run_with_retry",
                side_effect=lambda _name, fn, logger_instance=None: fn(),
            ),
        ):
            result = run_analytical_code(
                "summary_stats",
                '{"rows": [{"points": 10}], "field": "points"}',
            )

        self.assertEqual(result["status"], "partial_success")
        self.assertIn("division by zero", result["stderr"])

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
    def test_handles_unexpected_runtime_errors(self):
        with patch(
            "f1_agent.code_execution.vertexai.Client",
            side_effect=RuntimeError("boom"),
        ):
            result = run_analytical_code(
                "summary_stats",
                '{"rows": [{"points": 10}], "field": "points"}',
            )

        self.assertEqual(result["status"], "error")
        self.assertIn("unexpectedly", result["message"].lower())

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
