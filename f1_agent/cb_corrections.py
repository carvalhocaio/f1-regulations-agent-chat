"""Before/after-model callbacks for session correction tracking."""

from __future__ import annotations

import logging
import re

from f1_agent.cb_helpers import _extract_user_text, _prepend_user_context

logger = logging.getLogger(__name__)

_CORRECTIONS_KEY = "f1_corrections"
_MAX_CORRECTIONS = 20

# Patterns that indicate the user is correcting the agent
_CORRECTION_PATTERNS: list[re.Pattern] = [
    # Portuguese
    re.compile(
        r"(na verdade|na real|está errado|tá errado|errou|não foi|"
        r"faltou|esqueceu|incorreto|correto é|certo é|"
        r"mas foi|mas na verdade|mas o certo)",
        re.I,
    ),
    # English
    re.compile(
        r"(actually|that'?s wrong|that'?s incorrect|you'?re wrong|"
        r"not correct|incorrect|you missed|you forgot|"
        r"the correct.+is|it was actually|no,?\s+it was|"
        r"wrong.+it'?s|that'?s not right)",
        re.I,
    ),
]


def _is_correction(text: str) -> bool:
    """Return True if the message looks like a user correction."""
    return any(p.search(text) for p in _CORRECTION_PATTERNS)


def _get_corrections(callback_context) -> list[str]:
    """Read the corrections list from session state."""
    state = getattr(callback_context, "state", None)
    if state is None:
        return []
    try:
        return list(state.get(_CORRECTIONS_KEY, []))
    except Exception:
        return []


def _store_correction(callback_context, correction: str) -> None:
    """Append a correction to session state, capping at MAX_CORRECTIONS."""
    state = getattr(callback_context, "state", None)
    if state is None:
        return

    corrections = _get_corrections(callback_context)
    corrections.append(correction)

    # Keep only the most recent corrections
    if len(corrections) > _MAX_CORRECTIONS:
        corrections = corrections[-_MAX_CORRECTIONS:]

    state[_CORRECTIONS_KEY] = corrections


def inject_corrections(callback_context, llm_request):
    """Before-model callback: inject stored corrections into the prompt.

    If the user has previously corrected the agent in this session,
    prepend those corrections as user content so the model is aware of
    them without mutating the system instruction (preserving cache
    fingerprint stability).
    """
    corrections = _get_corrections(callback_context)
    if not corrections:
        return None

    corrections_text = "\n".join(f"- {c}" for c in corrections)
    addendum = (
        "## User corrections from this session\n"
        "The user has corrected you on the following points. "
        "Take these into account and do NOT repeat the same mistakes:\n"
        f"{corrections_text}"
    )

    _prepend_user_context(llm_request, addendum)
    logger.debug("Injected %d corrections into prompt", len(corrections))
    return None


def detect_corrections(callback_context, llm_response):
    """After-model callback: detect if the PREVIOUS user message was a correction.

    When the user corrects the agent, store the correction text in session
    state so it can be injected into future prompts via ``inject_corrections``.
    """
    del llm_response  # unused

    user_text = _extract_user_text(
        callback_context,
        type("_Stub", (), {"contents": []})(),
    )
    if not user_text:
        return None

    if not _is_correction(user_text):
        return None

    # Store the user's correction verbatim (trimmed)
    correction = user_text.strip()
    if len(correction) > 500:
        correction = correction[:500] + "…"

    _store_correction(callback_context, correction)
    logger.info("Correction stored: %s", correction[:80])

    return None  # don't modify the response
