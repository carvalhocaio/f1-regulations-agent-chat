import unittest

from f1_agent.agent import handle_rate_limit


class ModelErrorCallbackTests(unittest.TestCase):
    def test_accepts_error_keyword_and_handles_429(self):
        response = handle_rate_limit(
            callback_context=None,
            llm_request=None,
            error=Exception("429 Too Many Requests"),
        )

        self.assertIsNotNone(response)
        self.assertIn("per-minute request limit", response.content.parts[0].text)

    def test_accepts_legacy_exception_keyword(self):
        response = handle_rate_limit(
            callback_context=None,
            llm_request=None,
            exception=Exception("ResourceExhausted"),
        )

        self.assertIsNotNone(response)
        self.assertIn("per-minute request limit", response.content.parts[0].text)

    def test_handles_503_unavailable(self):
        response = handle_rate_limit(
            callback_context=None,
            llm_request=None,
            error=Exception("503 Service Unavailable"),
        )

        self.assertIsNotNone(response)
        self.assertIn("high demand", response.content.parts[0].text)

    def test_returns_none_for_other_errors(self):
        response = handle_rate_limit(
            callback_context=None,
            llm_request=None,
            error=Exception("Some other error"),
        )

        self.assertIsNone(response)


if __name__ == "__main__":
    unittest.main()
