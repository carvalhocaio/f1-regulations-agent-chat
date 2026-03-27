import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from f1_agent.callbacks import (
    apply_response_contract,
    get_structured_response_validation_counters,
    validate_structured_response,
)


class ResponseContractCallbackTests(unittest.TestCase):
    def _new_request(self):
        return SimpleNamespace(config=None, model="gemini-2.5-pro")

    @patch.dict(os.environ, {}, clear=False)
    def test_skips_when_contract_id_not_present(self):
        request = self._new_request()
        ctx = SimpleNamespace(state={})

        apply_response_contract(ctx, request)

        self.assertIsNone(request.config)

    @patch.dict(os.environ, {}, clear=False)
    def test_applies_sources_block_contract_from_state(self):
        request = self._new_request()
        ctx = SimpleNamespace(state={"response_contract_id": "sources_block_v1"})

        apply_response_contract(ctx, request)

        self.assertIsNotNone(request.config)
        self.assertEqual(request.config.response_mime_type, "application/json")
        self.assertEqual(request.config.response_schema["type"], "OBJECT")
        self.assertIn("sources", request.config.response_schema["properties"])

    @patch.dict(os.environ, {}, clear=False)
    def test_applies_contract_from_invocation_context_metadata(self):
        request = self._new_request()
        invocation_context = SimpleNamespace(
            metadata={"response_contract_id": "comparison_table_v1"}
        )
        ctx = SimpleNamespace(state={}, invocation_context=invocation_context)

        apply_response_contract(ctx, request)

        self.assertEqual(request.config.response_mime_type, "application/json")
        self.assertIn("columns", request.config.response_schema["properties"])
        self.assertIn("rows", request.config.response_schema["properties"])

    @patch.dict(os.environ, {"F1_STRUCTURED_RESPONSE_ENABLED": "false"}, clear=False)
    def test_respects_feature_flag(self):
        request = self._new_request()
        ctx = SimpleNamespace(state={"response_contract_id": "sources_block_v1"})

        apply_response_contract(ctx, request)

        self.assertIsNone(request.config)

    @patch.dict(os.environ, {}, clear=False)
    def test_ignores_unknown_contract_id(self):
        request = self._new_request()
        ctx = SimpleNamespace(state={"response_contract_id": "unknown_contract"})

        apply_response_contract(ctx, request)

        self.assertIsNone(request.config)


class StructuredResponseValidationTests(unittest.TestCase):
    def _new_response(self, text: str):
        return SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text=text)])
        )

    @patch.dict(os.environ, {}, clear=False)
    def test_validate_structured_response_success(self):
        ctx = SimpleNamespace(
            state={"f1_active_response_contract_id": "sources_block_v1"}
        )
        response = self._new_response(
            json.dumps(
                {
                    "schema_version": "v1",
                    "answer": "ok",
                    "sources": [],
                }
            )
        )

        validate_structured_response(ctx, response)

        counters = get_structured_response_validation_counters()
        self.assertGreaterEqual(counters.get("sources_block_v1:success", 0), 1)

    @patch.dict(os.environ, {}, clear=False)
    def test_validate_structured_response_invalid_json_replaced_with_fallback(self):
        ctx = SimpleNamespace(
            state={"f1_active_response_contract_id": "comparison_table_v1"}
        )
        response = self._new_response("not-json")

        validate_structured_response(ctx, response)

        parsed = json.loads(response.content.parts[0].text)
        self.assertEqual(parsed["schema_version"], "v1")
        self.assertIn("rows", parsed)
        counters = get_structured_response_validation_counters()
        self.assertGreaterEqual(counters.get("comparison_table_v1:parse_failure", 0), 1)

    @patch.dict(os.environ, {}, clear=False)
    def test_validate_structured_response_schema_failure_replaced_with_fallback(self):
        ctx = SimpleNamespace(
            state={"f1_active_response_contract_id": "sources_block_v1"}
        )
        response = self._new_response(
            json.dumps(
                {
                    "schema_version": "v1",
                    "answer": 123,
                    "sources": [],
                }
            )
        )

        validate_structured_response(ctx, response)

        parsed = json.loads(response.content.parts[0].text)
        self.assertEqual(parsed["schema_version"], "v1")
        self.assertIsInstance(parsed["answer"], str)
        counters = get_structured_response_validation_counters()
        self.assertGreaterEqual(counters.get("sources_block_v1:schema_failure", 0), 1)


if __name__ == "__main__":
    unittest.main()
