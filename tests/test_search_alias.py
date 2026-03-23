import unittest

from f1_agent.tools import search


class SearchAliasTests(unittest.TestCase):
    def test_search_alias_with_query_returns_guided_error(self):
        result = search(query="tyre compound rule in race")

        self.assertEqual(result["status"], "invalid_tool_alias")
        self.assertIn("valid_tools", result)
        self.assertIn("search_regulations", result["valid_tools"])
        self.assertIn("query_f1_history", result["valid_tools"])
        self.assertIn("google_search_agent", result["valid_tools"])
        self.assertEqual(result["suggested_query"], "tyre compound rule in race")

    def test_search_alias_with_request_returns_guided_error(self):
        result = search(request="latest drivers standings")

        self.assertEqual(result["status"], "invalid_tool_alias")
        self.assertEqual(result["suggested_query"], "latest drivers standings")
        self.assertIn("valid_tools", result)

    def test_search_alias_without_text_returns_guided_error(self):
        result = search()

        self.assertEqual(result["status"], "invalid_tool_alias")
        self.assertIn("valid_tools", result)
        self.assertNotIn("suggested_query", result)


if __name__ == "__main__":
    unittest.main()
