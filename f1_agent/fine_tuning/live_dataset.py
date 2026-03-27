"""Helpers for a live, anonymized fine-tuning dataset pipeline.

This module supports a continuous loop where real failures are collected,
sanitized, curated, and converted to Vertex AI SFT JSONL examples.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from f1_agent.fine_tuning.schema import build_example

_ALLOWED_CRITICALITY = {"high", "medium", "low"}
_MAX_PROMPT_LEN = 4000
_MAX_EXPECTED_LEN = 8000

_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,5}[\s.-]?\d{4}\b"
)
_LONG_ID_RE = re.compile(r"\b\d{8,}\b")
_SECRET_KV_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{8,}[\"']?"
)


def redact_text(text: str) -> str:
    """Best-effort redaction for common sensitive patterns."""
    if not text:
        return ""

    redacted = text
    redacted = _EMAIL_RE.sub("<REDACTED_EMAIL>", redacted)
    redacted = _URL_RE.sub("<REDACTED_URL>", redacted)
    redacted = _LONG_ID_RE.sub("<REDACTED_ID>", redacted)
    redacted = _PHONE_RE.sub("<REDACTED_PHONE>", redacted)
    redacted = _SECRET_KV_RE.sub("<REDACTED_SECRET>", redacted)
    return redacted


def _normalize_text(value: object, *, max_len: int, redact: bool) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    if redact:
        text = redact_text(text)
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _stable_id(prompt: str, expected_behavior: str, source: str) -> str:
    digest = hashlib.sha256(
        f"{source}\n{prompt}\n{expected_behavior}".encode()
    ).hexdigest()
    return f"live-{digest[:12]}"


def _sanitize_metadata(value: object, *, redact: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    sanitized: dict[str, Any] = {}
    for key, raw in value.items():
        safe_key = str(key)[:120]
        if isinstance(raw, str):
            text = raw.strip()
            if redact:
                text = redact_text(text)
            sanitized[safe_key] = text[:1000]
            continue
        if isinstance(raw, (int, float, bool)) or raw is None:
            sanitized[safe_key] = raw
            continue
        sanitized[safe_key] = str(raw)[:1000]
    return sanitized


def normalize_failure_case(
    payload: dict[str, Any], *, redact: bool = True
) -> dict[str, Any] | None:
    """Normalize one raw failure payload to the curated live-dataset schema."""
    prompt = _normalize_text(
        payload.get("prompt"), max_len=_MAX_PROMPT_LEN, redact=redact
    )
    expected = _normalize_text(
        payload.get("expected_behavior"), max_len=_MAX_EXPECTED_LEN, redact=redact
    )
    if not prompt or not expected:
        return None

    observed = _normalize_text(
        payload.get("observed_response"), max_len=_MAX_EXPECTED_LEN, redact=redact
    )
    category = (
        _normalize_text(payload.get("category"), max_len=120, redact=False) or "general"
    )
    failure_type = (
        _normalize_text(payload.get("failure_type"), max_len=120, redact=False)
        or "unknown"
    )
    source = (
        _normalize_text(payload.get("source"), max_len=120, redact=False) or "runtime"
    )
    criticality = (
        _normalize_text(payload.get("criticality"), max_len=20, redact=False).lower()
        or "medium"
    )
    if criticality not in _ALLOWED_CRITICALITY:
        criticality = "medium"

    case_id = _normalize_text(payload.get("id"), max_len=128, redact=False)
    if not case_id:
        case_id = _stable_id(prompt=prompt, expected_behavior=expected, source=source)

    ts = _normalize_text(payload.get("timestamp_utc"), max_len=64, redact=False)
    if not ts:
        ts = datetime.now(UTC).isoformat()

    return {
        "id": case_id,
        "prompt": prompt,
        "observed_response": observed,
        "expected_behavior": expected,
        "category": category,
        "criticality": criticality,
        "failure_type": failure_type,
        "source": source,
        "timestamp_utc": ts,
        "pii_redacted": bool(redact),
        "metadata": _sanitize_metadata(payload.get("metadata"), redact=redact),
    }


def normalize_failure_cases(
    raw_cases: list[dict[str, Any]], *, redact: bool = True
) -> list[dict[str, Any]]:
    """Normalize and deduplicate raw failure payloads."""
    curated: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for payload in raw_cases:
        case = normalize_failure_case(payload, redact=redact)
        if not case:
            continue
        dedupe_key = (
            f"{case['prompt']}\n{case['expected_behavior']}\n"
            f"{case['failure_type']}\n{case['category']}"
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        curated.append(case)

    return curated


def build_sft_examples_from_failures(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert curated failures into Vertex SFT training examples."""
    examples: list[dict[str, Any]] = []
    for case in cases:
        prompt = str(case.get("prompt", "")).strip()
        expected = str(case.get("expected_behavior", "")).strip()
        if not prompt or not expected:
            continue
        examples.append(
            build_example(
                user_message=prompt,
                model_answer=expected,
            )
        )
    return examples


def split_examples(
    examples: list[dict[str, Any]], *, test_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split examples into train/test."""
    if not examples:
        return [], []

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    test_size = int(len(shuffled) * test_ratio)
    if test_size <= 0:
        test_size = 1 if len(shuffled) > 1 else 0
    if test_size >= len(shuffled):
        test_size = max(1, len(shuffled) - 1)

    test = shuffled[:test_size]
    train = shuffled[test_size:]
    return train, test


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows as dictionaries."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary stats for reporting and manifests."""
    by_category = Counter(str(case.get("category", "general")) for case in cases)
    by_failure_type = Counter(
        str(case.get("failure_type", "unknown")) for case in cases
    )
    by_criticality = Counter(str(case.get("criticality", "medium")) for case in cases)

    return {
        "total_cases": len(cases),
        "by_category": dict(sorted(by_category.items())),
        "by_failure_type": dict(sorted(by_failure_type.items())),
        "by_criticality": dict(sorted(by_criticality.items())),
    }
