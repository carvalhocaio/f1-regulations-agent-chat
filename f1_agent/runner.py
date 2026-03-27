"""ADK runner wiring with managed Sessions support.

When running with Agent Engine env vars configured, this runner uses
``VertexAiSessionService`` to persist conversation context across calls.
Otherwise, it falls back to in-memory sessions for local dev.
"""

from __future__ import annotations

from google.adk.runners import Runner

from f1_agent.agent import build_app
from f1_agent.memory_bank import build_adk_memory_service
from f1_agent.sessions import build_adk_session_service

DEFAULT_APP_NAME = "f1_regulations_assistant"


def build_runner(app_name: str = DEFAULT_APP_NAME) -> Runner:
    """Build an ADK runner with session service configured by environment.

    Uses ``build_app()`` which wraps the root agent in an ``App`` with
    optional ``ContextCacheConfig`` for explicit Gemini context caching.
    """
    return Runner(
        app=build_app(),
        app_name=app_name,
        session_service=build_adk_session_service(),
        memory_service=build_adk_memory_service(),
        auto_create_session=True,
    )
