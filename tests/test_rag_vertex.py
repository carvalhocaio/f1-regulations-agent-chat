import unittest

from f1_agent.rag_vertex import _call_retrieval_query, _extract_contexts


class _NewStyleRagModule:
    class RagResource:
        def __init__(self, rag_corpus=None):
            self.rag_corpus = rag_corpus

    class Filter:
        def __init__(self, vector_distance_threshold=None):
            self.vector_distance_threshold = vector_distance_threshold

    class RagRetrievalConfig:
        def __init__(self, top_k=None, filter=None):
            self.top_k = top_k
            self.filter = filter

    @staticmethod
    def retrieval_query(text=None, rag_resources=None, rag_retrieval_config=None):
        return {
            "text": text,
            "rag_resources": rag_resources,
            "cfg": rag_retrieval_config,
        }


class _LegacyRagModule:
    @staticmethod
    def retrieval_query(corpus_names=None, text=None, similarity_top_k=None, **kwargs):
        if kwargs:
            raise TypeError("unexpected kwargs")
        return {
            "corpus_names": corpus_names,
            "text": text,
            "similarity_top_k": similarity_top_k,
        }


class RagVertexCallCompatibilityTests(unittest.TestCase):
    def test_call_retrieval_query_uses_new_style_signature(self):
        response = _call_retrieval_query(
            rag_module=_NewStyleRagModule,
            corpus_name="projects/p/locations/l/ragCorpora/1",
            query="drs",
            similarity_top_k=5,
            threshold=0.4,
        )

        self.assertEqual(response["text"], "drs")
        self.assertEqual(
            response["rag_resources"][0].rag_corpus,
            "projects/p/locations/l/ragCorpora/1",
        )
        self.assertEqual(response["cfg"].top_k, 5)
        self.assertEqual(response["cfg"].filter.vector_distance_threshold, 0.4)

    def test_call_retrieval_query_falls_back_to_legacy_signature(self):
        response = _call_retrieval_query(
            rag_module=_LegacyRagModule,
            corpus_name="projects/p/locations/l/ragCorpora/1",
            query="engine",
            similarity_top_k=3,
            threshold=None,
        )

        self.assertEqual(response["text"], "engine")
        self.assertEqual(response["similarity_top_k"], 3)
        self.assertEqual(
            response["corpus_names"], ["projects/p/locations/l/ragCorpora/1"]
        )

    def test_extract_contexts_handles_nested_contexts_container(self):
        response = type(
            "Resp",
            (),
            {
                "contexts": type(
                    "RagContexts",
                    (),
                    {"contexts": ["c1", "c2"]},
                )()
            },
        )()
        self.assertEqual(_extract_contexts(response), ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
