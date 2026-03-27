import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from f1_agent.callbacks import (
    apply_grounding_policy,
    get_grounding_validation_counters,
    validate_grounding_outcome,
)


class _FakeContent:
    def __init__(self, role, texts):
        self.role = role
        self.parts = [type("P", (), {"text": t})() for t in texts]


class GroundingPolicyTests(unittest.TestCase):
    def _ctx(self, user_text: str) -> SimpleNamespace:
        return SimpleNamespace(state={}, user_content=_FakeContent("user", [user_text]))

    def _req(self) -> SimpleNamespace:
        return SimpleNamespace(contents=[], config=None, model="gemini-2.5-pro")

    def _resp(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text=text)])
        )

    @patch.dict(os.environ, {"F1_GROUNDING_POLICY_ENABLED": "true"}, clear=False)
    def test_apply_grounding_policy_marks_time_sensitive_queries(self):
        ctx = self._ctx("Quem lidera o campeonato agora?")
        req = self._req()

        apply_grounding_policy(ctx, req)

        self.assertEqual(ctx.state.get("f1_grounding_policy"), "time_sensitive_public")
        self.assertTrue(ctx.state.get("f1_grounding_required"))
        self.assertEqual(ctx.state.get("f1_grounding_source"), "google")
        self.assertEqual(ctx.state.get("f1_response_contract_id"), "sources_block_v1")

    @patch.dict(os.environ, {"F1_GROUNDING_POLICY_ENABLED": "true"}, clear=False)
    def test_apply_grounding_policy_skips_historical_queries(self):
        ctx = self._ctx("Who won the 2023 championship?")
        req = self._req()

        apply_grounding_policy(ctx, req)

        self.assertEqual(ctx.state.get("f1_grounding_policy"), "non_critical")
        self.assertFalse(ctx.state.get("f1_grounding_required"))
        self.assertNotIn("f1_response_contract_id", ctx.state)

    @patch.dict(os.environ, {"F1_GROUNDING_POLICY_ENABLED": "true"}, clear=False)
    def test_validate_grounding_outcome_success_with_web_source(self):
        ctx = SimpleNamespace(
            state={
                "f1_grounding_policy": "time_sensitive_public",
                "f1_grounding_required": True,
                "f1_grounding_source": "google",
            }
        )
        response = self._resp(
            json.dumps(
                {
                    "schema_version": "v1",
                    "answer": "ok",
                    "sources": [
                        {
                            "source_type": "web",
                            "title": "FIA",
                            "reference": "fia.com",
                            "excerpt": "Updated standings.",
                            "url": "https://www.fia.com/",
                        }
                    ],
                }
            )
        )

        validate_grounding_outcome(ctx, response)

        counters = get_grounding_validation_counters()
        self.assertGreaterEqual(counters.get("time_sensitive_public:success", 0), 1)

    @patch.dict(
        os.environ,
        {
            "F1_GROUNDING_POLICY_ENABLED": "true",
            "F1_GROUNDING_POLICY_MODE": "observe",
        },
        clear=False,
    )
    def test_validate_grounding_outcome_observe_mode_does_not_override(self):
        ctx = SimpleNamespace(
            state={
                "f1_grounding_policy": "time_sensitive_public",
                "f1_grounding_required": True,
                "f1_grounding_source": "google",
            }
        )
        original = {
            "schema_version": "v1",
            "answer": "initial",
            "sources": [],
        }
        response = self._resp(json.dumps(original))

        validate_grounding_outcome(ctx, response)

        parsed = json.loads(response.content.parts[0].text)
        self.assertEqual(parsed["answer"], "initial")

    @patch.dict(
        os.environ,
        {
            "F1_GROUNDING_POLICY_ENABLED": "true",
            "F1_GROUNDING_POLICY_MODE": "enforce",
        },
        clear=False,
    )
    def test_validate_grounding_outcome_enforce_mode_overrides_when_missing(self):
        ctx = SimpleNamespace(
            state={
                "f1_grounding_policy": "time_sensitive_public",
                "f1_grounding_required": True,
                "f1_grounding_source": "google",
            }
        )
        response = self._resp(
            json.dumps(
                {
                    "schema_version": "v1",
                    "answer": "unverified answer",
                    "sources": [],
                }
            )
        )

        validate_grounding_outcome(ctx, response)

        parsed = json.loads(response.content.parts[0].text)
        self.assertIn("could not confidently verify", parsed["answer"])
        self.assertEqual(parsed["sources"], [])


if __name__ == "__main__":
    unittest.main()
