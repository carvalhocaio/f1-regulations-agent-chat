import unittest

from f1_agent.tools import (
    get_tool_validation_error_counters,
    query_f1_history,
    query_f1_history_template,
    search_regulations,
)


class ToolArgumentValidationTests(unittest.TestCase):
    def test_validation_errors_increment_observability_counter(self):
        before = get_tool_validation_error_counters().get(
            "query_f1_history:INVALID_QUERY", 0
        )

        result = query_f1_history("SELECT 1; SELECT 2")

        after = get_tool_validation_error_counters().get(
            "query_f1_history:INVALID_QUERY", 0
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(after, before + 1)

    def test_search_regulations_requires_non_empty_query(self):
        result = search_regulations("")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT")
        self.assertIn("query", result["message"])

    def test_query_template_rejects_non_object_params(self):
        result = query_f1_history_template(
            template_name="driver_champions",
            params='["not", "an", "object"]',
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT")
        self.assertIn("params", result["message"])

    def test_query_template_rejects_unknown_param_keys(self):
        result = query_f1_history_template(
            template_name="driver_champions",
            params='{"year": 2023, "foo": "bar"}',
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT")
        details = result["error"].get("details", {})
        self.assertIn("foo", details.get("unknown_keys", []))

    def test_query_history_rejects_multiple_statements(self):
        result = query_f1_history("SELECT 1; SELECT 2")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "INVALID_QUERY")
        self.assertIn("Multiple SQL statements", result["message"])


if __name__ == "__main__":
    unittest.main()
