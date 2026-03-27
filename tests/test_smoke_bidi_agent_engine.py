import importlib.util
import unittest
from pathlib import Path


def _load_smoke_bidi_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "deployment"
        / "smoke_bidi_agent_engine.py"
    )
    spec = importlib.util.spec_from_file_location(
        "smoke_bidi_agent_engine", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke_bidi = _load_smoke_bidi_module()


class StructuredSmokeValidationTests(unittest.TestCase):
    def test_validate_structured_output_accepts_valid_payload(self):
        payload_text = '{"schema_version":"v1","answer":"ok","sources":[]}'
        smoke_bidi._validate_structured_output(
            contract_id="sources_block_v1",
            response_text=payload_text,
        )

    def test_validate_structured_output_rejects_invalid_json(self):
        with self.assertRaises(RuntimeError):
            smoke_bidi._validate_structured_output(
                contract_id="sources_block_v1",
                response_text="not-json",
            )

    def test_validate_structured_output_rejects_schema_mismatch(self):
        invalid_payload = '{"schema_version":"v1","title":"x","columns":[],"rows":[]}'
        with self.assertRaises(RuntimeError):
            smoke_bidi._validate_structured_output(
                contract_id="sources_block_v1",
                response_text=invalid_payload,
            )


if __name__ == "__main__":
    unittest.main()
