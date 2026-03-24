import unittest

from f1_agent.callbacks import (
    FLASH_MODEL,
    _extract_user_text,
    _is_complex_question,
    route_model,
)


class _FakeContent:
    def __init__(self, role, texts):
        self.role = role
        self.parts = [type("P", (), {"text": t})() for t in texts]


class _FakeContext:
    def __init__(self, user_text=None):
        if user_text:
            self.user_content = _FakeContent("user", [user_text])
        else:
            self.user_content = None


class _FakeRequest:
    def __init__(self, model="gemini-2.5-pro", user_text=None):
        self.model = model
        if user_text:
            self.contents = [_FakeContent("user", [user_text])]
        else:
            self.contents = []


class ComplexityClassifierTests(unittest.TestCase):
    def test_simple_factual_question(self):
        self.assertFalse(_is_complex_question("Who won the 2023 championship?"))

    def test_simple_regulation_question(self):
        self.assertFalse(
            _is_complex_question("What does article 3.2 say about bodywork?")
        )

    def test_complex_comparison(self):
        self.assertTrue(
            _is_complex_question("Compare Verstappen vs Hamilton career stats")
        )

    def test_complex_temporal_range(self):
        self.assertTrue(_is_complex_question("Últimos 10 campeões mundiais"))

    def test_complex_evolution(self):
        self.assertTrue(
            _is_complex_question("How has the power unit evolved since 2014?")
        )

    def test_complex_why_question(self):
        self.assertTrue(
            _is_complex_question("Why did Ferrari lose the 2022 championship?")
        )

    def test_complex_regulation_plus_history(self):
        self.assertTrue(
            _is_complex_question("How do the 2026 regulations compare to past seasons?")
        )

    def test_complex_portuguese_comparison(self):
        self.assertTrue(_is_complex_question("Diferença entre o motor de 2024 e 2026"))


class ExtractUserTextTests(unittest.TestCase):
    def test_from_callback_context(self):
        ctx = _FakeContext("Hello F1!")
        req = _FakeRequest()
        self.assertEqual(_extract_user_text(ctx, req), "Hello F1!")

    def test_fallback_to_request_contents(self):
        ctx = _FakeContext()
        req = _FakeRequest(user_text="Who won?")
        self.assertEqual(_extract_user_text(ctx, req), "Who won?")

    def test_empty_when_no_content(self):
        ctx = _FakeContext()
        req = _FakeRequest()
        self.assertEqual(_extract_user_text(ctx, req), "")


class RouteModelTests(unittest.TestCase):
    def test_simple_query_routes_to_flash(self):
        ctx = _FakeContext("Who won the 2023 championship?")
        req = _FakeRequest()
        route_model(ctx, req)
        self.assertEqual(req.model, FLASH_MODEL)

    def test_complex_query_keeps_pro(self):
        ctx = _FakeContext("Compare Hamilton vs Verstappen career stats")
        req = _FakeRequest(model="gemini-2.5-pro")
        route_model(ctx, req)
        self.assertEqual(req.model, "gemini-2.5-pro")

    def test_empty_text_keeps_model(self):
        ctx = _FakeContext()
        req = _FakeRequest(model="gemini-2.5-pro")
        route_model(ctx, req)
        self.assertEqual(req.model, "gemini-2.5-pro")


if __name__ == "__main__":
    unittest.main()
