import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from f1_agent.tools import search_regulations


class RagBackendRoutingTests(unittest.TestCase):
    @patch.dict("os.environ", {"F1_RAG_BACKEND": "local"}, clear=False)
    @patch("f1_agent.tools._search_regulations_local")
    @patch("f1_agent.tools._search_regulations_vertex")
    @patch("f1_agent.tools._search_regulations_vector_search")
    def test_local_backend_uses_local_only(
        self, mock_vector_search, mock_vertex, mock_local
    ):
        mock_local.return_value = [
            Document(
                page_content="Article 1 content",
                metadata={"source": "s", "page": 1, "section": "A"},
            )
        ]

        result = search_regulations("engine rule")

        mock_local.assert_called_once()
        mock_vertex.assert_not_called()
        mock_vector_search.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["results"]), 1)

    @patch.dict("os.environ", {"F1_RAG_BACKEND": "vertex"}, clear=False)
    @patch("f1_agent.tools._search_regulations_local")
    @patch("f1_agent.tools._search_regulations_vertex")
    @patch("f1_agent.tools._search_regulations_vector_search")
    def test_vertex_backend_falls_back_to_local_when_empty(
        self, mock_vector_search, mock_vertex, mock_local
    ):
        mock_vertex.return_value = []
        mock_local.return_value = [
            Document(
                page_content="Fallback content",
                metadata={"source": "s", "page": 2, "section": "B"},
            )
        ]

        result = search_regulations("power unit")

        mock_vertex.assert_called_once()
        mock_local.assert_called_once()
        mock_vector_search.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"][0]["content"], "Fallback content")

    @patch.dict("os.environ", {"F1_RAG_BACKEND": "auto"}, clear=False)
    @patch("f1_agent.tools._search_regulations_local")
    @patch("f1_agent.tools._search_regulations_vertex")
    @patch("f1_agent.tools._search_regulations_vector_search")
    def test_auto_prefers_vector_search_then_vertex(
        self, mock_vector_search, mock_vertex, mock_local
    ):
        mock_vector_search.return_value = []
        mock_vertex.return_value = [
            Document(
                page_content="Vertex content",
                metadata={"source": "gs://docs/a.pdf", "page": 3, "section": "C"},
            )
        ]

        result = search_regulations("drs")

        mock_vector_search.assert_called_once()
        mock_vertex.assert_called_once()
        mock_local.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"][0]["content"], "Vertex content")

    @patch.dict("os.environ", {"F1_RAG_BACKEND": "vector_search"}, clear=False)
    @patch("f1_agent.tools._search_regulations_local")
    @patch("f1_agent.tools._search_regulations_vector_search")
    def test_vector_search_backend_falls_back_to_local_when_empty(
        self, mock_vector_search, mock_local
    ):
        mock_vector_search.return_value = []
        mock_local.return_value = [
            Document(
                page_content="Local fallback content",
                metadata={"source": "local", "page": 7, "section": "D"},
            )
        ]

        result = search_regulations("fuel flow")

        mock_vector_search.assert_called_once()
        mock_local.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"][0]["content"], "Local fallback content")


if __name__ == "__main__":
    unittest.main()
