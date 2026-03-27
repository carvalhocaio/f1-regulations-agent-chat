import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from f1_agent.callbacks import apply_throughput_request_type


class ThroughputRequestTypeCallbackTests(unittest.TestCase):
    def _new_request(self):
        return SimpleNamespace(config=None, model="gemini-2.5-pro")

    @patch.dict(os.environ, {}, clear=False)
    def test_defaults_to_shared(self):
        request = self._new_request()

        apply_throughput_request_type(None, request)

        self.assertIsNotNone(request.config)
        self.assertIsNotNone(request.config.http_options)
        self.assertEqual(
            request.config.http_options.headers["X-Vertex-AI-LLM-Request-Type"],
            "shared",
        )

    @patch.dict(os.environ, {"F1_VERTEX_LLM_REQUEST_TYPE": "dedicated"}, clear=False)
    def test_applies_dedicated_when_configured(self):
        request = self._new_request()

        apply_throughput_request_type(None, request)

        self.assertEqual(
            request.config.http_options.headers["X-Vertex-AI-LLM-Request-Type"],
            "dedicated",
        )

    @patch.dict(os.environ, {"F1_VERTEX_LLM_REQUEST_TYPE": "invalid"}, clear=False)
    def test_invalid_value_falls_back_to_shared(self):
        request = self._new_request()

        apply_throughput_request_type(None, request)

        self.assertEqual(
            request.config.http_options.headers["X-Vertex-AI-LLM-Request-Type"],
            "shared",
        )


if __name__ == "__main__":
    unittest.main()
