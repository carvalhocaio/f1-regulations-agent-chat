import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_smoke_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "deployment" / "smoke_agent_engine.py"
    )
    spec = importlib.util.spec_from_file_location("smoke_agent_engine", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke_module()


class ExtractSessionNameTests(unittest.TestCase):
    def test_prefers_nested_response_session_name_over_operation_name(self):
        expected = (
            "projects/p/locations/l/reasoningEngines/1234567890/sessions/session-1"
        )
        create_response = SimpleNamespace(
            name="projects/p/locations/l/operations/operation-1",
            response=SimpleNamespace(name=expected),
        )

        self.assertEqual(smoke._extract_session_name(create_response), expected)

    def test_accepts_direct_session_name(self):
        expected = (
            "projects/p/locations/l/reasoningEngines/1234567890/sessions/session-2"
        )
        create_response = {"name": expected}

        self.assertEqual(smoke._extract_session_name(create_response), expected)

    def test_rejects_operation_name_when_session_is_missing(self):
        create_response = SimpleNamespace(
            name="projects/p/locations/l/operations/operation-2",
            response=None,
        )

        self.assertIsNone(smoke._extract_session_name(create_response))


class SessionUserIdTests(unittest.TestCase):
    def test_reads_snake_case_user_id_attribute(self):
        session = SimpleNamespace(user_id="smoke-user")
        self.assertEqual(smoke._session_user_id(session), "smoke-user")

    def test_reads_camel_case_user_id_from_dict(self):
        session = {"userId": "smoke-user"}
        self.assertEqual(smoke._session_user_id(session), "smoke-user")


if __name__ == "__main__":
    unittest.main()
