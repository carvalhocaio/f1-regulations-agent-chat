"""Shared helpers used by multiple callback submodules."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from google.genai import types

logger = logging.getLogger(__name__)

_DB_MAX_YEAR = 2024


def _prepend_user_context(llm_request, text: str) -> None:
    """Insert dynamic context as a user content at the start of contents.

    By placing dynamic per-request information (temporal context, corrections,
    memories, examples) in user content instead of system_instruction, we keep
    the system instruction stable across requests.  This enables Gemini context
    caching — both implicit (automatic prefix dedup) and explicit (via ADK
    ContextCacheConfig) — since the cache fingerprint depends on a stable
    system_instruction.
    """
    llm_request.contents.insert(
        0,
        types.Content(role="user", parts=[types.Part(text=text)]),
    )


def _extract_user_text(callback_context, llm_request) -> str:
    """Extract the latest user message text from the request."""
    # Prefer callback_context.user_content (the original user message)
    user_content = getattr(callback_context, "user_content", None)
    if user_content and user_content.parts:
        texts = [p.text for p in user_content.parts if p.text]
        if texts:
            return " ".join(texts)

    # Fallback: scan llm_request.contents for the last user message
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            texts = [p.text for p in content.parts if p.text]
            if texts:
                return " ".join(texts)

    return ""


def _current_year() -> int:
    """Return current UTC year for temporal reasoning decisions."""
    return datetime.now(timezone.utc).year  # noqa: UP017 (py310 runtime)


def _current_date() -> date:
    """Return current UTC date for season-phase decisions."""
    return datetime.now(timezone.utc).date()  # noqa: UP017 (py310 runtime)
