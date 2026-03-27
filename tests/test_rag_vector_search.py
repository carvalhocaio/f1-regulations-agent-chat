import unittest

from f1_agent.rag_vector_search import _extract_metadata, _extract_text


class RagVectorSearchHelperTests(unittest.TestCase):
    def test_extract_text_prefers_data_fields_content(self):
        data_object = {
            "data_fields": {"content": "Regulation chunk"},
            "metadata_fields": {"text": "fallback"},
        }

        self.assertEqual(_extract_text(data_object), "Regulation chunk")

    def test_extract_metadata_merges_fields(self):
        data_object = {
            "data_fields": {"source": "gs://f1/doc.pdf", "page": 12},
            "metadata_fields": {"section": "C", "article": "3.1"},
        }

        metadata = _extract_metadata(data_object)
        self.assertEqual(metadata["source"], "gs://f1/doc.pdf")
        self.assertEqual(metadata["page"], "12")
        self.assertEqual(metadata["section"], "C")
        self.assertEqual(metadata["article"], "3.1")

    def test_extract_text_and_metadata_support_data_struct_shape(self):
        data_object = {
            "data": {
                "content": "Chunk from data struct",
                "source": "gs://f1/section_b.pdf",
                "page": "8",
                "section": "B",
                "article": "7.1.2",
            }
        }

        self.assertEqual(_extract_text(data_object), "Chunk from data struct")
        metadata = _extract_metadata(data_object)
        self.assertEqual(metadata["source"], "gs://f1/section_b.pdf")
        self.assertEqual(metadata["page"], "8")
        self.assertEqual(metadata["section"], "B")
        self.assertEqual(metadata["article"], "7.1.2")


if __name__ == "__main__":
    unittest.main()
