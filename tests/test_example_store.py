import os
import unittest
from unittest.mock import patch

from f1_agent.example_store import build_dynamic_examples_addendum


class _FakeExampleStore:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def search_examples(self, parameters, top_k):
        self.calls.append({"parameters": parameters, "top_k": top_k})
        return self.response


class DynamicExampleStoreTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_disabled_by_default(self):
        addendum, meta = build_dynamic_examples_addendum("Who won in 2023?")
        self.assertIsNone(addendum)
        self.assertFalse(meta["enabled"])

    @patch.dict(
        os.environ,
        {
            "F1_EXAMPLE_STORE_ENABLED": "true",
            "F1_EXAMPLE_STORE_NAME": "projects/p/locations/us-central1/exampleStores/s",
            "F1_EXAMPLE_STORE_TOP_K": "2",
            "F1_EXAMPLE_STORE_MIN_SCORE": "0.70",
        },
        clear=True,
    )
    @patch("f1_agent.example_store._get_example_store")
    def test_builds_addendum_from_search_results(self, mock_get_store):
        fake_store = _FakeExampleStore(
            {
                "results": [
                    {
                        "similarityScore": 0.92,
                        "example": {
                            "exampleId": "exampleTypes/stored_contents_example/examples/abc",
                            "storedContentsExample": {
                                "searchKey": "Quem foi campeao em 2023?",
                                "contentsExample": {
                                    "expectedContents": [
                                        {
                                            "content": {
                                                "parts": [
                                                    {
                                                        "functionCall": {
                                                            "name": "query_f1_history_template",
                                                            "args": {
                                                                "template_name": "driver_champions",
                                                                "params": '{"year": 2023}',
                                                            },
                                                        }
                                                    }
                                                ]
                                            }
                                        },
                                        {
                                            "content": {
                                                "role": "model",
                                                "parts": [
                                                    {
                                                        "text": "O campeao foi Max Verstappen.",
                                                    }
                                                ],
                                            }
                                        },
                                    ]
                                },
                            },
                        },
                    },
                    {
                        "similarityScore": 0.50,
                        "example": {
                            "exampleId": "low-score",
                            "storedContentsExample": {
                                "contentsExample": {"expectedContents": []}
                            },
                        },
                    },
                ]
            }
        )
        mock_get_store.return_value = fake_store

        addendum, meta = build_dynamic_examples_addendum("Quem ganhou em 2023?")

        self.assertIsNotNone(addendum)
        self.assertIn("Dynamic few-shot examples", addendum)
        self.assertIn("query_f1_history_template", addendum)
        self.assertIn("O campeao foi Max Verstappen", addendum)
        self.assertEqual(meta["example_count"], 1)
        self.assertEqual(meta["top_similarity"], 0.92)
        self.assertEqual(fake_store.calls[0]["top_k"], 2)

    @patch.dict(
        os.environ,
        {
            "F1_EXAMPLE_STORE_ENABLED": "true",
            "F1_EXAMPLE_STORE_NAME": "",
        },
        clear=True,
    )
    def test_enabled_without_store_name_skips(self):
        addendum, meta = build_dynamic_examples_addendum("Who is leading now?")
        self.assertIsNone(addendum)
        self.assertTrue(meta["enabled"])
        self.assertFalse(meta["store_configured"])


if __name__ == "__main__":
    unittest.main()
