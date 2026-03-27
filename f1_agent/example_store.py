"""Dynamic few-shot retrieval from Vertex AI Example Store.

This module retrieves examples similar to the current user query and converts
them into a compact instruction addendum that can be injected into the prompt.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_RESOURCE_NAME_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/exampleStores/(?P<id>[^/]+)$"
)

_ENABLED_ENV = "F1_EXAMPLE_STORE_ENABLED"
_STORE_NAME_ENV = "F1_EXAMPLE_STORE_NAME"
_TOP_K_ENV = "F1_EXAMPLE_STORE_TOP_K"
_MIN_SCORE_ENV = "F1_EXAMPLE_STORE_MIN_SCORE"
_MAX_EXAMPLES_ENV = "F1_EXAMPLE_STORE_MAX_EXAMPLES"
_MAX_CHARS_ENV = "F1_EXAMPLE_STORE_MAX_CHARS"

_DEFAULT_TOP_K = 3
_DEFAULT_MIN_SCORE = 0.65
_DEFAULT_MAX_EXAMPLES = 3
_DEFAULT_MAX_CHARS = 3200

_store_cache: dict[str, Any] = {}
_store_init_failed: set[str] = set()


@dataclass(frozen=True)
class ExampleStoreSettings:
    enabled: bool
    store_name: str
    top_k: int
    min_score: float
    max_examples: int
    max_chars: int


@dataclass(frozen=True)
class RetrievedExample:
    example_id: str
    similarity_score: float
    search_key: str
    expected_lines: list[str]


def load_settings() -> ExampleStoreSettings:
    """Load dynamic few-shot settings from environment variables."""
    return ExampleStoreSettings(
        enabled=_env_bool(_ENABLED_ENV, default=False),
        store_name=os.environ.get(_STORE_NAME_ENV, "").strip(),
        top_k=max(1, _env_int(_TOP_K_ENV, _DEFAULT_TOP_K)),
        min_score=max(0.0, min(1.0, _env_float(_MIN_SCORE_ENV, _DEFAULT_MIN_SCORE))),
        max_examples=max(1, _env_int(_MAX_EXAMPLES_ENV, _DEFAULT_MAX_EXAMPLES)),
        max_chars=max(500, _env_int(_MAX_CHARS_ENV, _DEFAULT_MAX_CHARS)),
    )


def build_dynamic_examples_addendum(
    user_text: str,
) -> tuple[str | None, dict[str, Any]]:
    """Build an instruction addendum with relevant examples for ``user_text``."""
    settings = load_settings()
    metadata: dict[str, Any] = {
        "enabled": settings.enabled,
        "store_configured": bool(settings.store_name),
        "requested_top_k": settings.top_k,
        "example_count": 0,
        "top_similarity": None,
        "store_name": settings.store_name,
    }

    if not settings.enabled:
        return None, metadata

    if not settings.store_name:
        logger.warning(
            "%s=true but %s is empty; skipping dynamic examples",
            _ENABLED_ENV,
            _STORE_NAME_ENV,
        )
        return None, metadata

    question = (user_text or "").strip()
    if not question:
        return None, metadata

    example_store = _get_example_store(settings.store_name)
    if example_store is None:
        return None, metadata

    try:
        response = example_store.search_examples(
            parameters={"stored_contents_example_key": question},
            top_k=settings.top_k,
        )
    except Exception:
        logger.warning(
            "Example Store search failed; continuing without examples", exc_info=True
        )
        return None, metadata

    examples = _extract_examples(response, min_score=settings.min_score)
    if not examples:
        return None, metadata

    selected = examples[: settings.max_examples]
    metadata["example_count"] = len(selected)
    metadata["top_similarity"] = selected[0].similarity_score
    metadata["example_ids"] = [item.example_id for item in selected]

    addendum = _format_addendum(selected, max_chars=settings.max_chars)
    return addendum, metadata


def _get_example_store(store_name: str):
    if store_name in _store_cache:
        return _store_cache[store_name]

    if store_name in _store_init_failed:
        return None

    try:
        import vertexai
        from vertexai.preview import example_stores

        match = _RESOURCE_NAME_RE.match(store_name)
        if match:
            vertexai.init(
                project=match.group("project"),
                location=match.group("location"),
            )

        store = example_stores.ExampleStore(store_name)
    except Exception:
        _store_init_failed.add(store_name)
        logger.warning("Example Store initialization failed", exc_info=True)
        return None

    _store_cache[store_name] = store
    return store


def _extract_examples(response: Any, min_score: float) -> list[RetrievedExample]:
    raw_results = _field(response, "results") or []
    examples: list[RetrievedExample] = []

    for item in raw_results:
        score = _field(item, "similarityScore", "similarity_score")
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        if score_value < min_score:
            continue

        item_example = _field(item, "example")
        stored = _field(
            item_example, "storedContentsExample", "stored_contents_example"
        )
        if stored is None:
            continue

        contents_example = _field(stored, "contentsExample", "contents_example")
        expected_contents = _field(
            contents_example, "expectedContents", "expected_contents"
        )
        lines = _summarize_expected_contents(expected_contents or [])
        if not lines:
            continue

        example_id = _field(item_example, "exampleId", "example_id") or "unknown"
        search_key = _field(stored, "searchKey", "search_key") or ""

        examples.append(
            RetrievedExample(
                example_id=str(example_id),
                similarity_score=score_value,
                search_key=str(search_key),
                expected_lines=lines,
            )
        )

    examples.sort(key=lambda e: e.similarity_score, reverse=True)
    return examples


def _summarize_expected_contents(expected_contents: list[Any]) -> list[str]:
    lines: list[str] = []

    for expected in expected_contents:
        content = _field(expected, "content")
        if not content:
            continue

        parts = _field(content, "parts") or []
        for part in parts:
            text = _field(part, "text")
            if text:
                lines.append(f"Model answer: {_truncate(str(text), 180)}")

            function_call = _field(part, "functionCall", "function_call")
            if function_call:
                name = _field(function_call, "name") or "unknown_tool"
                args = _field(function_call, "args") or {}
                lines.append(
                    f"Call `{name}` with args {_truncate(_to_json(args), 180)}"
                )

            function_response = _field(part, "functionResponse", "function_response")
            if function_response:
                name = _field(function_response, "name") or "unknown_tool"
                response = _field(function_response, "response") or {}
                lines.append(
                    f"Use `{name}` response {_truncate(_to_json(response), 180)}"
                )

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    return deduped


def _format_addendum(examples: list[RetrievedExample], max_chars: int) -> str:
    lines = [
        "\n\n## Dynamic few-shot examples from real corrected errors",
        "Use these examples only when the current question is semantically similar.",
    ]

    for index, example in enumerate(examples, start=1):
        lines.append(
            f"\n### Example {index} (similarity={example.similarity_score:.2f})"
        )
        if example.search_key:
            lines.append(f"User: {example.search_key}")
        lines.append("Expected behavior:")
        for item in example.expected_lines[:5]:
            lines.append(f"- {item}")
        lines.append(f"- Example ID: {example.example_id}")

    addendum = "\n".join(lines)
    if len(addendum) <= max_chars:
        return addendum

    suffix = "\n- [dynamic examples truncated to fit prompt budget]"
    return addendum[: max(0, max_chars - len(suffix))] + suffix


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


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return str(value)


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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        logger.warning("Invalid float in %s=%r; using default %.2f", name, raw, default)
        return default
