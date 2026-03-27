"""Compatibility layer between local test stubs and real Google ADK classes.

When ``google.adk`` is installed (for example, while running ``adk web``),
this module re-exports the real ADK types so ``root_agent`` is a true
``BaseAgent`` instance.

When ADK is not installed, it falls back to lightweight local dataclasses used
by tests and local non-ADK flows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

_USING_REAL_ADK = False

try:  # pragma: no cover - exercised only in environments with google.adk
    from google.adk import Runner as Runner
    from google.adk import Agent as Agent
    from google.adk.apps import App as App
    from google.adk.models import Gemini as Gemini
    from google.adk.models.llm_response import LlmResponse as LlmResponse
    from google.adk.sessions import (
        InMemorySessionService as InMemorySessionService,
    )
    from google.adk.sessions import Session as Session

    _USING_REAL_ADK = True
except Exception:  # pragma: no cover - normal path in CI without ADK
    @dataclass
    class LlmResponse:
        content: types.Content

    @dataclass
    class Gemini:
        model: str
        retry_options: Any = None

    @dataclass
    class Agent:
        name: str
        model: Any
        description: str
        static_instruction: str
        tools: list[Any] = field(default_factory=list)
        before_model_callback: list[Any] = field(default_factory=list)
        after_model_callback: list[Any] = field(default_factory=list)
        on_model_error_callback: Any = None

    @dataclass
    class App:
        name: str
        root_agent: Agent

    @dataclass
    class Session:
        app_name: str
        user_id: str
        id: str
        state: dict[str, Any] = field(default_factory=dict)

        @property
        def session_id(self) -> str:
            return self.id

    class InMemorySessionService:
        def __init__(self):
            self._sessions: dict[tuple[str, str, str], Session] = {}

        def create_session(
            self,
            *,
            app_name: str,
            user_id: str,
            session_id: str | None = None,
            state: dict[str, Any] | None = None,
            config: dict[str, Any] | None = None,
        ) -> Session:
            del config
            sid = session_id or str(uuid.uuid4())
            session = Session(
                app_name=app_name,
                user_id=user_id,
                id=sid,
                state=dict(state or {}),
            )
            self._sessions[(app_name, user_id, sid)] = session
            return session

        def get_session(
            self, *, app_name: str, user_id: str, session_id: str
        ) -> Session | None:
            return self._sessions.get((app_name, user_id, session_id))

        def list_sessions(self, *, app_name: str, user_id: str) -> list[Session]:
            return [
                value
                for (session_app, session_user, _), value in self._sessions.items()
                if session_app == app_name and session_user == user_id
            ]

        def delete_session(
            self, *, app_name: str, user_id: str, session_id: str
        ) -> None:
            self._sessions.pop((app_name, user_id, session_id), None)

    @dataclass
    class Runner:
        app: App
        app_name: str
        session_service: InMemorySessionService | None = None
        memory_service: Any = None
        auto_create_session: bool = True

