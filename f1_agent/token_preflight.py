"""Token preflight check for context window management.

Counts tokens via the Gemini CountTokens API before each model call and
progressively truncates injected context blocks when the request exceeds
a configurable threshold.  Designed as the last before-model callback so
that all dynamic context (temporal, corrections, memories, examples) and
model routing have already been applied.

Disabled by default — enable via ``F1_PREFLIGHT_TOKEN_CHECK_ENABLED=true``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from f1_agent.env_utils import env_bool, env_float, env_int

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
}
_DEFAULT_CONTEXT_WINDOW = 1_048_576
_DEFAULT_MODEL = "gemini-2.5-pro"

# Headers injected by before-model callbacks — used to identify removable
# context blocks.  Order defines *truncation priority* (first = removed first).
_INJECTED_HEADERS: list[tuple[str, str]] = [
    ("## Dynamic few-shot examples", "examples"),
    ("## Long-term user memory", "memories"),
    ("## User corrections from this session", "corrections"),
    ("## Runtime temporal context", "temporal"),
]

# ---------------------------------------------------------------------------
# Lazy genai client singleton
# ---------------------------------------------------------------------------

_client = None
_client_lock = threading.Lock()


def _get_client():
    """Return a shared ``genai.Client``, created on first use (thread-safe)."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            from google import genai  # noqa: PLC0415

            _client = genai.Client()
    return _client


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_request_tokens(
    model: str,
    contents: list,
    system_instruction: str | None = None,
    tools: list | None = None,
) -> int:
    """Call the CountTokens API and return total token count.

    Uses ``resilience.run_with_retry`` for transient-error resilience.
    """
    from google.genai import types  # noqa: PLC0415

    from f1_agent.resilience import run_with_retry

    config_kwargs: dict = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if tools:
        config_kwargs["tools"] = tools

    config = types.CountTokensConfig(**config_kwargs) if config_kwargs else None

    def _call():
        client = _get_client()
        resp = client.models.count_tokens(
            model=model,
            contents=contents,
            config=config,
        )
        return resp.total_tokens

    return run_with_retry(
        "count_tokens",
        _call,
        logger_instance=logger,
    )


# ---------------------------------------------------------------------------
# Block identification
# ---------------------------------------------------------------------------


def _identify_injected_blocks(contents: list) -> list[tuple[int, str]]:
    """Return ``(index, category)`` pairs for injected context blocks.

    Scans ``contents`` for user-role items whose text starts with a known
    header.  Results are ordered by truncation priority (least → most
    critical).
    """
    found: list[tuple[int, str]] = []

    for idx, item in enumerate(contents):
        role = getattr(item, "role", None)
        if role != "user":
            continue

        parts = getattr(item, "parts", None)
        if not parts:
            continue

        text = getattr(parts[0], "text", "") or ""
        for header, category in _INJECTED_HEADERS:
            if header in text:
                found.append((idx, category))
                break

    # Sort by _INJECTED_HEADERS order (already scanned in priority order,
    # but ensure deterministic priority via explicit sort).
    priority = {cat: i for i, (_, cat) in enumerate(_INJECTED_HEADERS)}
    found.sort(key=lambda pair: priority.get(pair[1], 999))

    return found


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------


def _compute_threshold(model: str, threshold_fraction: float, hard_limit: int) -> int:
    """Return the effective token limit for the given model."""
    if hard_limit > 0:
        return hard_limit
    window = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)
    return int(window * threshold_fraction)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PreflightResult:
    """Structured result of the preflight check."""

    model: str
    original_tokens: int
    final_tokens: int
    threshold: int
    removed: list[str] = field(default_factory=list)
    truncated: bool = False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def check_and_truncate(llm_request) -> PreflightResult | None:
    """Count tokens and progressively truncate injected blocks if over budget.

    Returns a ``PreflightResult`` describing what happened, or ``None`` if
    the feature is disabled.
    """
    if not env_bool("F1_PREFLIGHT_TOKEN_CHECK_ENABLED", False):
        return None

    model = getattr(llm_request, "model", None) or _DEFAULT_MODEL
    # Normalise Gemini model wrapper to string
    model_str = getattr(model, "model", None) or str(model)

    threshold_fraction = env_float("F1_PREFLIGHT_TOKEN_THRESHOLD", 0.80)
    hard_limit = env_int("F1_PREFLIGHT_TOKEN_HARD_LIMIT", 0)
    threshold = _compute_threshold(model_str, threshold_fraction, hard_limit)

    # Extract system instruction text if present.
    sys_instruction = None
    config = getattr(llm_request, "config", None)
    if config:
        si = getattr(config, "system_instruction", None)
        if si:
            # May be a string or Content object.
            sys_instruction = getattr(si, "text", None) or str(si)

    total = count_request_tokens(
        model=model_str,
        contents=list(llm_request.contents),
        system_instruction=sys_instruction,
    )

    result = PreflightResult(
        model=model_str,
        original_tokens=total,
        final_tokens=total,
        threshold=threshold,
    )

    if total <= threshold:
        logger.info(
            "preflight_tokens | model=%s tokens=%d threshold=%d status=ok",
            model_str,
            total,
            threshold,
        )
        return result

    # Over threshold — progressively remove injected blocks.
    logger.warning(
        "preflight_tokens | model=%s tokens=%d threshold=%d status=over — "
        "starting progressive truncation",
        model_str,
        total,
        threshold,
    )

    blocks = _identify_injected_blocks(llm_request.contents)

    for idx, category in blocks:
        # Remove the block.  Use the *current* index — after prior removals
        # indices shift, so we search by object identity.
        target = None
        for i, item in enumerate(llm_request.contents):
            parts = getattr(item, "parts", None)
            if not parts:
                continue
            text = getattr(parts[0], "text", "") or ""
            header = next((h for h, c in _INJECTED_HEADERS if c == category), None)
            if header and header in text:
                target = i
                break

        if target is None:
            continue

        llm_request.contents.pop(target)
        result.removed.append(category)
        result.truncated = True

        total = count_request_tokens(
            model=model_str,
            contents=list(llm_request.contents),
            system_instruction=sys_instruction,
        )
        result.final_tokens = total

        logger.info(
            "preflight_truncate | removed=%s tokens_now=%d threshold=%d",
            category,
            total,
            threshold,
        )

        if total <= threshold:
            break

    if result.final_tokens > threshold:
        logger.warning(
            "preflight_tokens | all injected blocks removed but still over "
            "threshold — tokens=%d threshold=%d (conversation history too large)",
            result.final_tokens,
            threshold,
        )

    logger.info(
        "preflight_tokens | model=%s original=%d final=%d threshold=%d "
        "removed=%s truncated=%s",
        result.model,
        result.original_tokens,
        result.final_tokens,
        result.threshold,
        result.removed,
        result.truncated,
    )

    return result
