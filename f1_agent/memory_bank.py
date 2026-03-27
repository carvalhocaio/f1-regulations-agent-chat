"""Long-term user memory integration for Agent Engine Memory Bank.

This module provides two runtime capabilities:
- Retrieve relevant memories for a user and inject them into prompt context.
- Trigger memory generation from Agent Engine session history.

All behavior is feature-flagged and fails open.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import vertexai
from google.api_core import exceptions as gcp_exceptions

logger = logging.getLogger(__name__)

_ENABLED_ENV = "F1_MEMORY_BANK_ENABLED"
_PROJECT_ENV = "F1_MEMORY_BANK_PROJECT_ID"
_LOCATION_ENV = "F1_MEMORY_BANK_LOCATION"
_AGENT_ENGINE_NAME_ENV = "F1_MEMORY_BANK_AGENT_ENGINE_NAME"
_AGENT_ENGINE_ID_ENV = "GOOGLE_CLOUD_AGENT_ENGINE_ID"

_MAX_FACTS_ENV = "F1_MEMORY_BANK_MAX_FACTS"
_MAX_FACT_CHARS_ENV = "F1_MEMORY_BANK_MAX_FACT_CHARS"
_FETCH_LIMIT_ENV = "F1_MEMORY_BANK_FETCH_LIMIT"
_GENERATE_CORRECTION_ONLY_ENV = "F1_MEMORY_BANK_GENERATE_ON_CORRECTION_ONLY"

_DEFAULT_LOCATION = "us-central1"
_DEFAULT_MAX_FACTS = 5
_DEFAULT_MAX_FACT_CHARS = 240
_DEFAULT_FETCH_LIMIT = 20


@dataclass(frozen=True)
class MemoryBankSettings:
    enabled: bool
    project_id: str
    location: str
    agent_engine_name: str
    max_facts: int
    max_fact_chars: int
    fetch_limit: int
    generate_on_correction_only: bool


def load_settings() -> MemoryBankSettings:
    project_id = (
        os.environ.get(_PROJECT_ENV, "").strip()
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.environ.get("F1_RAG_PROJECT_ID", "").strip()
    )
    location = (
        os.environ.get(_LOCATION_ENV, "").strip()
        or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        or _DEFAULT_LOCATION
    )
    explicit_name = os.environ.get(_AGENT_ENGINE_NAME_ENV, "").strip()
    agent_engine_name = explicit_name or _agent_engine_name_from_env(
        project_id, location
    )

    return MemoryBankSettings(
        enabled=_env_bool(_ENABLED_ENV, default=False),
        project_id=project_id,
        location=location,
        agent_engine_name=agent_engine_name,
        max_facts=max(1, _env_int(_MAX_FACTS_ENV, _DEFAULT_MAX_FACTS)),
        max_fact_chars=max(80, _env_int(_MAX_FACT_CHARS_ENV, _DEFAULT_MAX_FACT_CHARS)),
        fetch_limit=max(1, _env_int(_FETCH_LIMIT_ENV, _DEFAULT_FETCH_LIMIT)),
        generate_on_correction_only=_env_bool(
            _GENERATE_CORRECTION_ONLY_ENV,
            default=True,
        ),
    )


def build_memory_addendum(
    user_id: str, query_text: str
) -> tuple[str | None, dict[str, Any]]:
    settings = load_settings()
    metadata: dict[str, Any] = {
        "enabled": settings.enabled,
        "configured": bool(settings.project_id and settings.agent_engine_name),
        "memory_count": 0,
    }

    if not settings.enabled:
        return None, metadata

    if not user_id.strip():
        return None, metadata

    if not settings.project_id or not settings.agent_engine_name:
        logger.debug("Memory Bank configured incompletely; skipping injection")
        return None, metadata

    memories = _retrieve_user_memories(settings, user_id=user_id, query_text=query_text)
    if not memories:
        return None, metadata

    selected = memories[: settings.max_facts]
    metadata["memory_count"] = len(selected)
    addendum_lines = [
        "\n\n## Long-term user memory (cross-session)",
        "Use only if relevant to the current request. Do not overfit.",
    ]
    addendum_lines.extend(f"- {item}" for item in selected)
    return "\n".join(addendum_lines), metadata


def generate_memories_from_session(
    session_name: str,
    user_id: str | None,
    *,
    wait_for_completion: bool = False,
) -> bool:
    settings = load_settings()
    if not settings.enabled:
        return False

    if not session_name or "/sessions/" not in session_name:
        return False

    if not settings.project_id or not settings.agent_engine_name:
        logger.debug("Memory Bank configured incompletely; skipping generation")
        return False

    try:
        client = vertexai.Client(
            project=settings.project_id, location=settings.location
        )
        kwargs: dict[str, Any] = {
            "name": settings.agent_engine_name,
            "vertex_session_source": {"session": session_name},
            "config": {"wait_for_completion": wait_for_completion},
        }
        if user_id and user_id.strip():
            kwargs["scope"] = {"user_id": user_id.strip()}

        client.agent_engines.memories.generate(**kwargs)
        return True
    except (
        gcp_exceptions.PermissionDenied,
        gcp_exceptions.InvalidArgument,
        gcp_exceptions.InternalServerError,
        gcp_exceptions.ServiceUnavailable,
        gcp_exceptions.DeadlineExceeded,
    ):
        logger.warning("Memory generation failed", exc_info=True)
        return False


def build_adk_memory_service():
    """Return VertexAiMemoryBankService when enabled and configured.

    This service is optional for the Runner. Returns None when disabled or if
    the SDK feature is unavailable.
    """
    settings = load_settings()
    if not settings.enabled:
        return None

    agent_engine_id = os.environ.get(_AGENT_ENGINE_ID_ENV, "").strip()
    if not agent_engine_id:
        logger.debug("%s is missing; ADK memory service disabled", _AGENT_ENGINE_ID_ENV)
        return None

    try:
        from google.adk.memory import VertexAiMemoryBankService
    except Exception:
        logger.warning("VertexAiMemoryBankService is unavailable", exc_info=True)
        return None

    try:
        return VertexAiMemoryBankService(
            project=settings.project_id,
            location=settings.location,
            agent_engine_id=agent_engine_id,
        )
    except TypeError:
        # Express mode variants may accept fewer arguments.
        return VertexAiMemoryBankService(agent_engine_id=agent_engine_id)
    except Exception:
        logger.warning("Failed to initialize ADK memory service", exc_info=True)
        return None


def _retrieve_user_memories(
    settings: MemoryBankSettings,
    *,
    user_id: str,
    query_text: str,
) -> list[str]:
    try:
        client = vertexai.Client(
            project=settings.project_id, location=settings.location
        )

        # Query support can vary by SDK version. Attempt with query first,
        # then fall back to scope-only retrieval.
        try:
            records = list(
                client.agent_engines.memories.retrieve(
                    name=settings.agent_engine_name,
                    scope={"user_id": user_id},
                    query=query_text,
                    max_results=settings.fetch_limit,
                )
            )
        except TypeError:
            records = list(
                client.agent_engines.memories.retrieve(
                    name=settings.agent_engine_name,
                    scope={"user_id": user_id},
                )
            )
    except (
        gcp_exceptions.PermissionDenied,
        gcp_exceptions.InvalidArgument,
        gcp_exceptions.InternalServerError,
        gcp_exceptions.ServiceUnavailable,
        gcp_exceptions.DeadlineExceeded,
    ):
        logger.warning("Memory retrieval failed", exc_info=True)
        return []

    memories: list[str] = []
    seen: set[str] = set()
    for entry in records[: settings.fetch_limit]:
        text = _extract_memory_text(entry)
        if not text:
            continue

        compact = " ".join(text.split())
        compact = _truncate(compact, settings.max_fact_chars)
        if compact in seen:
            continue
        seen.add(compact)
        memories.append(compact)

    return memories


def _extract_memory_text(entry: Any) -> str | None:
    direct = _field(entry, "fact", "text", "memory", "content")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    nested = _field(entry, "memory")
    if isinstance(nested, dict):
        value = nested.get("fact") or nested.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _agent_engine_name_from_env(project_id: str, location: str) -> str:
    engine_id = os.environ.get(_AGENT_ENGINE_ID_ENV, "").strip()
    if not engine_id:
        return ""
    if engine_id.startswith("projects/"):
        return engine_id
    if not project_id:
        return ""
    return f"projects/{project_id}/locations/{location}/reasoningEngines/{engine_id}"


def _field(value: Any, *keys: str) -> Any:
    if value is None:
        return None
    for key in keys:
        if isinstance(value, dict) and key in value:
            return value[key]
        candidate = getattr(value, key, None)
        if candidate is not None:
            return candidate
    return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        logger.warning("Invalid integer in %s=%r; using default %d", name, raw, default)
        return default
