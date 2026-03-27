"""Runner wiring with local in-memory session support."""

from __future__ import annotations

from f1_agent.adk_compat import Runner
from f1_agent.agent import build_app
from f1_agent.memory_bank import build_adk_memory_service
from f1_agent.sessions import build_adk_session_service

DEFAULT_APP_NAME = "f1_regulations_assistant"


def build_runner(app_name: str = DEFAULT_APP_NAME) -> Runner:
    """Build a runner configured with local session and memory services."""
    return Runner(
        app=build_app(),
        app_name=app_name,
        session_service=build_adk_session_service(),
        memory_service=build_adk_memory_service(),
        auto_create_session=True,
    )
