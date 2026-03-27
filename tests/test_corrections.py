import unittest

from f1_agent.callbacks import (
    _CORRECTIONS_KEY,
    _get_corrections,
    _is_correction,
    _store_correction,
    detect_corrections,
    inject_corrections,
)


class _FakeState(dict):
    """Dict that behaves like ADK session state."""

    pass


class _FakeContent:
    def __init__(self, role, texts):
        self.role = role
        self.parts = [type("P", (), {"text": t})() for t in texts]


class _FakeContext:
    def __init__(self, user_text=None, state=None):
        if user_text:
            self.user_content = _FakeContent("user", [user_text])
        else:
            self.user_content = None
        self.state = state if state is not None else _FakeState()


class _FakeRequest:
    def __init__(self):
        self.contents = []
        self.config = type("C", (), {"system_instruction": "Base instruction"})()

    def append_instructions(self, instructions):
        for inst in instructions:
            if isinstance(self.config.system_instruction, str):
                self.config.system_instruction += "\n\n" + inst
        return []


class _FakeResponse:
    def __init__(self, text=None):
        if text:
            self.content = type(
                "C",
                (),
                {"parts": [type("P", (), {"text": text})()]},
            )()
        else:
            self.content = None


class CorrectionDetectionTests(unittest.TestCase):
    def test_portuguese_na_verdade(self):
        self.assertTrue(_is_correction("Na verdade, Hamilton ganhou 7 títulos"))

    def test_portuguese_errou(self):
        self.assertTrue(_is_correction("Você errou, não foi em 2020"))

    def test_portuguese_faltou(self):
        self.assertTrue(_is_correction("Faltou mencionar a temporada de 2019"))

    def test_english_actually(self):
        self.assertTrue(_is_correction("Actually, Verstappen won in 2021"))

    def test_english_thats_wrong(self):
        self.assertTrue(_is_correction("That's wrong, it was Alonso"))

    def test_english_you_missed(self):
        self.assertTrue(_is_correction("You missed the 2022 season"))

    def test_not_a_correction(self):
        self.assertFalse(_is_correction("Who won the 2023 championship?"))

    def test_not_a_correction_simple(self):
        self.assertFalse(_is_correction("Tell me about Hamilton's career"))


class CorrectionStorageTests(unittest.TestCase):
    def test_store_and_retrieve(self):
        ctx = _FakeContext(state=_FakeState())
        _store_correction(ctx, "Hamilton has 7 titles, not 6")
        corrections = _get_corrections(ctx)
        self.assertEqual(len(corrections), 1)
        self.assertIn("Hamilton has 7 titles", corrections[0])

    def test_max_corrections_cap(self):
        ctx = _FakeContext(state=_FakeState())
        for i in range(25):
            _store_correction(ctx, f"Correction {i}")
        corrections = _get_corrections(ctx)
        self.assertEqual(len(corrections), 20)
        # Should keep the most recent
        self.assertEqual(corrections[-1], "Correction 24")
        self.assertEqual(corrections[0], "Correction 5")

    def test_empty_state_returns_empty(self):
        ctx = _FakeContext(state=_FakeState())
        self.assertEqual(_get_corrections(ctx), [])


class InjectCorrectionsTests(unittest.TestCase):
    def test_no_corrections_no_change(self):
        ctx = _FakeContext(state=_FakeState())
        req = _FakeRequest()
        inject_corrections(ctx, req)
        self.assertEqual(len(req.contents), 0)

    def test_corrections_prepended_to_contents(self):
        state = _FakeState()
        state[_CORRECTIONS_KEY] = ["Hamilton has 7 titles"]
        ctx = _FakeContext(state=state)
        req = _FakeRequest()
        inject_corrections(ctx, req)
        # Corrections are now injected as user content (not system instruction)
        self.assertEqual(len(req.contents), 1)
        injected_text = req.contents[0].parts[0].text
        self.assertIn("Hamilton has 7 titles", injected_text)
        self.assertIn("User corrections", injected_text)


class DetectCorrectionsCallbackTests(unittest.TestCase):
    def test_correction_detected_and_stored(self):
        state = _FakeState()
        ctx = _FakeContext(
            user_text="Na verdade, Hamilton ganhou 7 títulos",
            state=state,
        )
        resp = _FakeResponse("Ok, you are right!")
        detect_corrections(ctx, resp)
        corrections = _get_corrections(ctx)
        self.assertEqual(len(corrections), 1)

    def test_normal_message_not_stored(self):
        state = _FakeState()
        ctx = _FakeContext(
            user_text="Who won the 2023 championship?",
            state=state,
        )
        resp = _FakeResponse("Verstappen won")
        detect_corrections(ctx, resp)
        corrections = _get_corrections(ctx)
        self.assertEqual(len(corrections), 0)


if __name__ == "__main__":
    unittest.main()
