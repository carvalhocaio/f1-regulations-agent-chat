import unittest

from f1_agent.rag import _extract_article, _tokenize


class TokenizeTests(unittest.TestCase):
    def test_basic_tokenization(self):
        tokens = _tokenize("The car must not exceed 2000mm.")
        self.assertEqual(tokens, ["the", "car", "must", "not", "exceed", "2000mm"])

    def test_removes_punctuation(self):
        tokens = _tokenize("Article 3.2.1 — Bodywork (dimensions)")
        self.assertIn("article", tokens)
        self.assertIn("3", tokens)
        self.assertIn("bodywork", tokens)
        self.assertIn("dimensions", tokens)

    def test_lowercases(self):
        tokens = _tokenize("DRS Drag Reduction System")
        self.assertEqual(tokens, ["drs", "drag", "reduction", "system"])


class ExtractArticleTests(unittest.TestCase):
    def test_extracts_article_number(self):
        text = "According to Article 3.2.1, the bodywork dimensions..."
        self.assertEqual(_extract_article(text), "3.2.1")

    def test_extracts_two_level_article(self):
        text = "As per 12.4, the financial cap..."
        self.assertEqual(_extract_article(text), "12.4")

    def test_no_article_returns_empty(self):
        text = "General overview of the regulations"
        self.assertEqual(_extract_article(text), "")

    def test_extracts_first_match(self):
        text = "Articles 3.1 and 3.2 discuss bodywork"
        self.assertEqual(_extract_article(text), "3.1")

    def test_three_level_article(self):
        text = "Rule 5.2.3.1 defines power unit limits"
        self.assertEqual(_extract_article(text), "5.2.3.1")


if __name__ == "__main__":
    unittest.main()
