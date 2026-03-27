"""Collect raw live failure cases from eval and runtime artifacts.

This script creates a JSONL file that feeds the live fine-tuning dataset
builder. It focuses on automation-friendly sources:

- evaluation dataset + eval report + eval gate result
- runtime event/log JSONL (optional)
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TOOL_VALIDATION_RE = re.compile(
    r"tool_validation_error\s*\|\s*tool=(?P<tool>[^\s]+)\s+code=(?P<code>[^\s]+)",
    re.IGNORECASE,
)
_STRUCTURED_RESPONSE_RE = re.compile(
    r"structured_response_validation\s*\|\s*contract_id=(?P<contract>[^\s]+)\s+outcome=(?P<outcome>[^\s]+)",
    re.IGNORECASE,
)
_GROUNDING_RE = re.compile(
    r"grounding_validation\s*\|\s*policy=(?P<policy>[^\s]+).*outcome=(?P<outcome>[^\s]+)",
    re.IGNORECASE,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _expected_behavior_from_category(row: dict[str, Any]) -> str:
    category = str(row.get("category", "general")).strip().lower()
    reference = str(row.get("reference", "")).strip()
    if reference:
        return f"Answer should align with this reference: {reference}"

    if category == "tool_routing":
        return (
            "Use only declared tools with valid names and arguments, then provide "
            "a direct factual answer."
        )
    if category == "response_format":
        return (
            "Return the requested structured output format and keep the payload "
            "parser-compatible."
        )
    if category == "regulation_grounding":
        return "Ground regulation claims with precise article/section pointers in the final response."
    if category == "temporal_reasoning":
        return "Resolve relative temporal expressions explicitly and avoid stale-year assumptions."
    if category == "safety":
        return "Follow safety constraints and reject requests to fabricate or bypass policy."
    return "Provide a precise and verifiable answer with correct tool usage."


def collect_from_eval_artifacts(
    *,
    eval_dataset_rows: list[dict[str, Any]],
    eval_gate_result: dict[str, Any] | None,
    eval_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Create raw failures from eval artifacts.

    The gate output determines whether we collect all rows or target rows by
    impacted metric families.
    """
    failures: list[dict[str, Any]] = []

    failed_metrics: set[str] = set()
    if isinstance(eval_gate_result, dict):
        checks = eval_gate_result.get("checks")
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict):
                    continue
                metric = str(check.get("metric", "")).strip().upper()
                passed = bool(check.get("passed", True))
                if metric and not passed:
                    failed_metrics.add(metric)

    summary_scores: dict[str, Any] = {}
    if isinstance(eval_report, dict):
        payload = eval_report.get("summary_metrics")
        if isinstance(payload, dict):
            summary_scores = payload

    for row in eval_dataset_rows:
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        category = str(row.get("category", "general")).strip() or "general"
        criticality = str(row.get("criticality", "medium")).strip().lower() or "medium"

        include = False
        failure_type = "quality_regression"
        if not failed_metrics:
            include = True
        elif "TOOL_USE_QUALITY" in failed_metrics and category in {
            "tool_routing",
            "response_format",
        }:
            include = True
            failure_type = "tool_or_format_regression"
        elif "FINAL_RESPONSE_QUALITY" in failed_metrics and category in {
            "factuality",
            "historical_fact",
            "regulation_grounding",
            "temporal_reasoning",
            "response_format",
        }:
            include = True
            failure_type = "final_response_regression"

        if not include:
            continue

        failures.append(
            {
                "id": str(row.get("id", "")).strip() or "",
                "prompt": prompt,
                "expected_behavior": _expected_behavior_from_category(row),
                "observed_response": (
                    "Eval regression candidate. summary_metrics="
                    + json.dumps(summary_scores, ensure_ascii=True)
                ),
                "category": category,
                "criticality": criticality,
                "failure_type": failure_type,
                "source": "eval_pipeline",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "metadata": {
                    "case_id": str(row.get("id", "")).strip(),
                    "failed_metrics": sorted(failed_metrics),
                },
            }
        )

    return failures


def _extract_prompt_from_event(event: dict[str, Any]) -> str:
    for key in ("prompt", "user_prompt", "query", "message", "input"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_text_runtime_event(text: str) -> dict[str, Any] | None:
    tool_match = _TOOL_VALIDATION_RE.search(text)
    if tool_match:
        return {
            "failure_type": "tool_validation_error",
            "category": "tool_routing",
            "expected_behavior": (
                "Use declared tool names with valid arguments and avoid validation errors."
            ),
            "metadata": {
                "tool_name": tool_match.group("tool"),
                "error_code": tool_match.group("code"),
            },
        }

    structured_match = _STRUCTURED_RESPONSE_RE.search(text)
    if structured_match:
        return {
            "failure_type": "structured_response_validation",
            "category": "response_format",
            "expected_behavior": (
                "Return schema-compliant structured output when a response contract is active."
            ),
            "metadata": {
                "contract_id": structured_match.group("contract"),
                "outcome": structured_match.group("outcome"),
            },
        }

    grounding_match = _GROUNDING_RE.search(text)
    if grounding_match and grounding_match.group("outcome").lower() == "missing":
        return {
            "failure_type": "grounding_validation_missing",
            "category": "regulation_grounding",
            "expected_behavior": (
                "For time-sensitive queries, include grounded web evidence before final answer."
            ),
            "metadata": {
                "policy": grounding_match.group("policy"),
                "outcome": grounding_match.group("outcome"),
            },
        }

    return None


def collect_from_runtime_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect raw failures from runtime event/log rows."""
    failures: list[dict[str, Any]] = []

    for event in events:
        text = ""
        if isinstance(event.get("text"), str):
            text = str(event.get("text"))
        elif isinstance(event.get("message"), str):
            text = str(event.get("message"))
        elif isinstance(event.get("log"), str):
            text = str(event.get("log"))

        parsed = _parse_text_runtime_event(text)
        if not parsed:
            continue

        prompt = _extract_prompt_from_event(event)
        if not prompt:
            prompt = "Runtime validation event without captured prompt"

        criticality = (
            str(event.get("criticality", "medium")).strip().lower() or "medium"
        )
        failures.append(
            {
                "id": str(event.get("id", "")).strip() or "",
                "prompt": prompt,
                "expected_behavior": parsed["expected_behavior"],
                "observed_response": text[:4000],
                "category": parsed["category"],
                "criticality": criticality,
                "failure_type": parsed["failure_type"],
                "source": "runtime_logs",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "metadata": {
                    **parsed.get("metadata", {}),
                    "raw_event_keys": sorted(str(k) for k in event.keys()),
                },
            }
        )

    return failures


def _dedupe_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = (
            f"{row.get('prompt', '')}\n{row.get('expected_behavior', '')}\n"
            f"{row.get('failure_type', '')}\n{row.get('source', '')}"
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect raw live fine-tuning failures"
    )
    parser.add_argument("--eval-dataset-file", default="")
    parser.add_argument("--eval-report-file", default="")
    parser.add_argument("--eval-gate-result-file", default="")
    parser.add_argument(
        "--runtime-events-file",
        action="append",
        default=[],
        help="JSONL runtime events/log file (can be repeated)",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-failures", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    eval_dataset_rows: list[dict[str, Any]] = []
    eval_report: dict[str, Any] | None = None
    eval_gate_result: dict[str, Any] | None = None

    if args.eval_dataset_file:
        eval_dataset_path = Path(args.eval_dataset_file)
        if eval_dataset_path.exists():
            eval_dataset_rows = _load_jsonl(eval_dataset_path)

    if args.eval_report_file:
        eval_report_path = Path(args.eval_report_file)
        if eval_report_path.exists():
            eval_report = _load_json(eval_report_path)

    if args.eval_gate_result_file:
        eval_gate_path = Path(args.eval_gate_result_file)
        if eval_gate_path.exists():
            eval_gate_result = _load_json(eval_gate_path)

    failures: list[dict[str, Any]] = []
    if eval_dataset_rows:
        failures.extend(
            collect_from_eval_artifacts(
                eval_dataset_rows=eval_dataset_rows,
                eval_gate_result=eval_gate_result,
                eval_report=eval_report,
            )
        )

    runtime_events: list[dict[str, Any]] = []
    for event_file in args.runtime_events_file:
        path = Path(event_file)
        if path.exists():
            runtime_events.extend(_load_jsonl(path))
    if runtime_events:
        failures.extend(collect_from_runtime_events(runtime_events))

    failures = _dedupe_failures(failures)
    if args.max_failures > 0:
        failures = failures[: args.max_failures]

    _write_jsonl(Path(args.output_file), failures)
    print(
        json.dumps(
            {
                "output_file": args.output_file,
                "total_failures": len(failures),
                "sources": {
                    "eval_pipeline": sum(
                        1 for row in failures if row.get("source") == "eval_pipeline"
                    ),
                    "runtime_logs": sum(
                        1 for row in failures if row.get("source") == "runtime_logs"
                    ),
                },
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
