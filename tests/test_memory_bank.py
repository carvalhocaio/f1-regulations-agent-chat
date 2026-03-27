import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from f1_agent.callbacks import (
    detect_corrections,
    inject_long_term_memories,
    sync_memory_bank,
)
from f1_agent.memory_bank import build_memory_addendum, generate_memories_from_session


class _FakeRequest:
    def __init__(self):
        self.contents = []
        self.instructions = []

    def append_instructions(self, values):
        self.instructions.extend(values)


class MemoryBankAddendumTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_disabled_by_default(self):
        addendum, meta = build_memory_addendum("user-1", "query")
        self.assertIsNone(addendum)
        self.assertFalse(meta["enabled"])

    @patch.dict(
        os.environ,
        {
            "F1_MEMORY_BANK_ENABLED": "true",
            "F1_MEMORY_BANK_PROJECT_ID": "my-project",
            "F1_MEMORY_BANK_LOCATION": "us-central1",
            "F1_MEMORY_BANK_AGENT_ENGINE_NAME": "projects/p/locations/us-central1/reasoningEngines/r",
            "F1_MEMORY_BANK_MAX_FACTS": "2",
        },
        clear=True,
    )
    @patch("f1_agent.memory_bank.vertexai.Client")
    def test_builds_addendum_from_retrieved_memories(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.agent_engines.memories.retrieve.return_value = [
            {"fact": "User prefers concise answers."},
            {"fact": "User follows McLaren closely."},
            {"fact": "Extra memory"},
        ]

        addendum, meta = build_memory_addendum("user-1", "How is Piastri doing?")

        self.assertIsNotNone(addendum)
        self.assertIn("Long-term user memory", addendum)
        self.assertIn("User prefers concise answers", addendum)
        self.assertEqual(meta["memory_count"], 2)


class MemoryBankGenerationTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "F1_MEMORY_BANK_ENABLED": "true",
            "F1_MEMORY_BANK_PROJECT_ID": "my-project",
            "F1_MEMORY_BANK_LOCATION": "us-central1",
            "F1_MEMORY_BANK_AGENT_ENGINE_NAME": "projects/p/locations/us-central1/reasoningEngines/r",
        },
        clear=True,
    )
    @patch("f1_agent.memory_bank.vertexai.Client")
    def test_generates_memories_from_session(self, mock_client_cls):
        ok = generate_memories_from_session(
            "projects/p/locations/us-central1/reasoningEngines/r/sessions/s1",
            "user-1",
        )
        self.assertTrue(ok)

        mock_client = mock_client_cls.return_value
        mock_client.agent_engines.memories.generate.assert_called_once()


class MemoryBankCallbackTests(unittest.TestCase):
    @patch("f1_agent.callbacks.build_memory_addendum")
    def test_inject_long_term_memories_appends_instructions(self, mock_build):
        mock_build.return_value = (
            "\n\n## Long-term user memory\n- User prefers concise answers.",
            {"enabled": True, "configured": True, "memory_count": 1},
        )
        ctx = SimpleNamespace(
            user_id="user-1",
            state={},
            user_content=SimpleNamespace(parts=[SimpleNamespace(text="Question")]),
        )
        req = _FakeRequest()

        inject_long_term_memories(ctx, req)

        # Memories are now injected as user content (not system instruction)
        self.assertEqual(len(req.contents), 1)
        self.assertIn("Long-term user memory", req.contents[0].parts[0].text)

    @patch.dict(os.environ, {"F1_MEMORY_BANK_ENABLED": "true"}, clear=False)
    @patch("f1_agent.callbacks.generate_memories_from_session")
    @patch("f1_agent.callbacks.load_memory_bank_settings")
    def test_sync_memory_bank_runs_after_correction(
        self,
        mock_settings,
        mock_generate,
    ):
        mock_settings.return_value = SimpleNamespace(
            enabled=True,
            generate_on_correction_only=True,
            project_id="my-project",
            location="us-central1",
            agent_engine_name="projects/p/locations/us-central1/reasoningEngines/r",
        )

        ctx = SimpleNamespace(
            user_id="user-1",
            session_id="session-1",
            state={},
            user_content=SimpleNamespace(
                parts=[SimpleNamespace(text="Na verdade, voce errou")]
            ),
        )

        detect_corrections(ctx, None)
        sync_memory_bank(ctx, None)

        self.assertTrue(mock_generate.called)


if __name__ == "__main__":
    unittest.main()
