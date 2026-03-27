"""Lightweight local compatibility layer replacing Google ADK runtime types.

This project previously depended on ``google-adk`` for agent/session runtime
objects. The dependency chain now excludes ADK, so this module provides the
small subset of classes used by the codebase and tests.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from google.genai import types


@dataclass
class LlmResponse:
    content: types.Content


@dataclass
class Gemini:
    model: str
    retry_options: Any = None


@dataclass
class ContextCacheConfig:
    cache_intervals: int = 10
    ttl_seconds: int = 1800
    min_tokens: int = 4096


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
    context_cache_config: ContextCacheConfig | None = None


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

    def delete_session(self, *, app_name: str, user_id: str, session_id: str) -> None:
        self._sessions.pop((app_name, user_id, session_id), None)


@dataclass
class Runner:
    app: App
    app_name: str
    session_service: InMemorySessionService | None = None
    memory_service: Any = None
    auto_create_session: bool = True


class GoogleSearchTool:
    def __init__(self, bypass_multi_tools_limit: bool = True):
        del bypass_multi_tools_limit
        self.name = "google_search"

    def __call__(self, request: str, **kwargs) -> dict[str, Any]:
        del kwargs
        return {
            "status": "disabled",
            "request": request,
            "reason": (
                "google_search tool is disabled in local runtime after removing "
                "google-adk dependency."
            ),
        }
