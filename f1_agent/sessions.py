"""Session helpers for managed conversational context.

This module centralizes:
- Anonymous/stable ``user_id`` normalization (for no-login clients)
- Session TTL config creation
- Local in-memory session service selection
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass

from google.adk.sessions import InMemorySessionService

logger = logging.getLogger(__name__)

_MAX_USER_ID_LEN = 128
_SAFE_USER_ID_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class SessionIdentity:
    """Normalized session identity used across the client/runtime boundary."""

    user_id: str
    session_id: str | None = None


def _sanitize_user_id(raw: str) -> str:
    cleaned = _SAFE_USER_ID_RE.sub("-", raw.strip())
    cleaned = cleaned.strip("-._")
    if not cleaned:
        raise ValueError("user_id cannot be empty after sanitization")
    return cleaned[:_MAX_USER_ID_LEN]


def anonymous_user_id(client_id: str) -> str:
    """Build a deterministic anonymous user id from a stable client id.

    The caller should persist ``client_id`` in browser local storage/cookie.
    """
    if not client_id or not client_id.strip():
        raise ValueError("client_id is required to build anonymous user_id")

    digest = hashlib.sha256(client_id.strip().encode("utf-8")).hexdigest()[:24]
    return f"anon-{digest}"


def resolve_user_id(*, user_id: str | None, client_id: str | None) -> str:
    """Resolve the canonical ``user_id`` for session operations.

    Priority:
    1) Explicit ``user_id`` (sanitized)
    2) Deterministic anonymous id from ``client_id``
    """
    if user_id and user_id.strip():
        return _sanitize_user_id(user_id)
    if client_id and client_id.strip():
        return anonymous_user_id(client_id)
    raise ValueError("Provide user_id or client_id for session operations")


def build_session_identity(
    *,
    user_id: str | None,
    session_id: str | None,
    client_id: str | None,
) -> SessionIdentity:
    """Return normalized identity that can be propagated across calls."""
    normalized_session_id = (
        session_id.strip() if session_id and session_id.strip() else None
    )
    return SessionIdentity(
        user_id=resolve_user_id(user_id=user_id, client_id=client_id),
        session_id=normalized_session_id,
    )


def session_ttl_config(ttl_seconds: int | None) -> dict[str, str] | None:
    """Build the session config payload for TTL.

    Returns ``None`` if no TTL is provided.
    """
    if ttl_seconds is None:
        return None
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be > 0")
    return {"ttl": f"{ttl_seconds}s"}


def build_adk_session_service():
    """Return local in-memory session service.

    Managed session backends were removed to eliminate Cloud Spanner-linked
    dependencies from the runtime stack.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    agent_engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID")

    if project and location and agent_engine_id:
        logger.info(
            "Managed session env vars detected but ignored; using in-memory "
            "session service to keep runtime free of Spanner dependencies."
        )

    return InMemorySessionService()
