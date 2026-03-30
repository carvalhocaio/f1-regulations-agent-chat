import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from f1_agent.tools import search_regulations


class RagBackendRoutingTests(unittest.TestCase):
    @patch.dict("os.environ", {"F1_RAG_BACKEND": "local"}, clear=False)
    @patch("f1_agent.tools_rag._search_regulations_local")
    def test_local_backend_uses_local_only(self, mock_local):
        mock_local.return_value = [
            Document(
                page_content="Article 1 content",
                metadata={"source": "s", "page": 1, "section": "A"},
            )
        ]

        result = search_regulations("engine rule")

        mock_local.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["results"]), 1)

    @patch.dict("os.environ", {"F1_RAG_BACKEND": "auto"}, clear=False)
    @patch("f1_agent.tools_rag._search_regulations_local")
    def test_auto_backend_is_forced_to_local(self, mock_local):
        mock_local.return_value = [
            Document(
                page_content="Fallback content",
                metadata={"source": "s", "page": 2, "section": "B"},
            )
        ]

        result = search_regulations("power unit")

        mock_local.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"][0]["content"], "Fallback content")


if __name__ == "__main__":
    unittest.main()
