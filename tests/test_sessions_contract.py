import unittest
from unittest import mock

from google.adk.sessions import InMemorySessionService

from f1_agent.sessions import (
    anonymous_user_id,
    build_adk_session_service,
    build_session_identity,
    resolve_user_id,
    session_ttl_config,
)


class UserIdResolutionTests(unittest.TestCase):
    def test_prefers_explicit_user_id(self):
        user_id = resolve_user_id(user_id=" User.123 ", client_id="abc")
        self.assertEqual(user_id, "User.123")

    def test_anonymous_user_id_is_deterministic(self):
        a = anonymous_user_id("browser-installation-id")
        b = anonymous_user_id("browser-installation-id")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("anon-"))

    def test_requires_identifier(self):
        with self.assertRaises(ValueError):
            resolve_user_id(user_id=None, client_id=None)

    def test_build_session_identity_normalizes_session_id(self):
        identity = build_session_identity(
            user_id="pilot",
            session_id=" 12345 ",
            client_id=None,
        )
        self.assertEqual(identity.user_id, "pilot")
        self.assertEqual(identity.session_id, "12345")


class SessionTtlTests(unittest.TestCase):
    def test_none_ttl_returns_none(self):
        self.assertIsNone(session_ttl_config(None))

    def test_positive_ttl_builds_payload(self):
        self.assertEqual(session_ttl_config(3600), {"ttl": "3600s"})

    def test_invalid_ttl_raises(self):
        with self.assertRaises(ValueError):
            session_ttl_config(0)


class SessionServiceSelectionTests(unittest.TestCase):
    def test_falls_back_to_in_memory_when_env_missing(self):
        with mock.patch.dict(
            "os.environ",
            {
                "GOOGLE_CLOUD_PROJECT": "",
                "GOOGLE_CLOUD_LOCATION": "",
                "GOOGLE_CLOUD_AGENT_ENGINE_ID": "",
            },
            clear=False,
        ):
            service = build_adk_session_service()
            self.assertIsInstance(service, InMemorySessionService)

    def test_keeps_in_memory_service_when_env_is_set(self):
        with mock.patch.dict(
            "os.environ",
            {
                "GOOGLE_CLOUD_PROJECT": "my-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
                "GOOGLE_CLOUD_AGENT_ENGINE_ID": "1234567890",
            },
            clear=False,
        ):
            service = build_adk_session_service()
            self.assertIsInstance(service, InMemorySessionService)


if __name__ == "__main__":
    unittest.main()
