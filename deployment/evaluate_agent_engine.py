"""Run Gen AI Evaluation against a deployed Agent Engine resource."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Number
from pathlib import Path
from typing import Any

import vertexai
from vertexai import types

DEFAULT_METRICS = [
    "FINAL_RESPONSE_QUALITY",
    "TOOL_USE_QUALITY",
    "HALLUCINATION",
    "SAFETY",
]


@dataclass
class EvalCase:
    case_id: str
    prompt: str
    category: str
    criticality: str
    reference: str | None = None


def _load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"Missing prompt at {path}:{line_number}")
            case_id = str(payload.get("id", f"case-{line_number:04d}")).strip()
            category = str(payload.get("category", "general")).strip() or "general"
            criticality = str(payload.get("criticality", "normal")).strip() or "normal"
            reference_raw = payload.get("reference")
            reference = str(reference_raw).strip() if reference_raw else None
            cases.append(
                EvalCase(
                    case_id=case_id,
                    prompt=prompt,
                    category=category,
                    criticality=criticality,
                    reference=reference,
                )
            )

    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def _resolve_rubric_metrics(metric_names: list[str]) -> list[types.RubricMetric]:
    resolved: list[types.RubricMetric] = []
    for name in metric_names:
        normalized = name.strip().upper()
        if not normalized:
            continue
        try:
            resolved.append(getattr(types.RubricMetric, normalized))
        except AttributeError as exc:
            raise ValueError(f"Unsupported rubric metric: {name}") from exc
    if not resolved:
        raise ValueError("At least one rubric metric is required")
    return resolved


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, Number)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_plain(model_dump())
    as_dict = getattr(value, "__dict__", None)
    if isinstance(as_dict, dict):
        return _to_plain(as_dict)
    return str(value)


def _extract_score(payload: Any) -> float | None:
    if isinstance(payload, Number):
        return float(payload)
    if not isinstance(payload, dict):
        return None

    for key in ("score", "mean", "value", "avg", "average"):
        candidate = payload.get(key)
        if isinstance(candidate, Number):
            return float(candidate)
    for candidate in payload.values():
        if isinstance(candidate, Number):
            return float(candidate)
    return None


def _search_metric_score(payload: Any, metric_name: str) -> float | None:
    metric_key = metric_name.upper()
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key).upper()
            if key_text == metric_key or metric_key in key_text:
                score = _extract_score(value)
                if score is not None:
                    return score
                nested = _search_metric_score(value, metric_name)
                if nested is not None:
                    return nested
            nested = _search_metric_score(value, metric_name)
            if nested is not None:
                return nested
    if isinstance(payload, list):
        for item in payload:
            nested = _search_metric_score(item, metric_name)
            if nested is not None:
                return nested
    return None


def _extract_summary_scores(
    summary_payload: Any, metric_names: list[str]
) -> dict[str, float | None]:
    return {
        metric_name: _search_metric_score(summary_payload, metric_name)
        for metric_name in metric_names
    }


def _run_name(evaluation_run: Any) -> str | None:
    for field in ("name", "resource_name", "resourceName"):
        value = getattr(evaluation_run, field, None)
        if value:
            return str(value)
    return None


def _run_state(evaluation_run: Any) -> str:
    for field in ("state", "status"):
        value = getattr(evaluation_run, field, None)
        if value is None:
            continue
        state_text = str(value).strip()
        if state_text:
            return state_text
    return "UNKNOWN"


def _wait_for_run(
    *,
    client: vertexai.Client,
    run_name: str,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> Any:
    started = time.monotonic()
    while True:
        run = client.evals.get_evaluation_run(name=run_name)
        state = _run_state(run).upper()
        if any(token in state for token in ("SUCCEEDED", "COMPLETED", "DONE")):
            return run
        if any(token in state for token in ("FAILED", "CANCELLED")):
            raise RuntimeError(f"Evaluation run ended with state={state}")
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(
                f"Timed out while waiting for evaluation run: {run_name}"
            )
        time.sleep(max(1.0, poll_interval_seconds))


def _build_release_metadata() -> dict[str, str]:
    return {
        "git_sha": os.getenv("GITHUB_SHA", ""),
        "git_ref": os.getenv("GITHUB_REF", ""),
        "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deployed Agent Engine quality with rubric metrics"
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--agent-resource-name", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--output-file", default="eval_report.json")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated RubricMetric names",
    )
    parser.add_argument(
        "--dest",
        default="",
        help="Optional GCS destination prefix for evaluation outputs",
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = _load_cases(Path(args.dataset_file))
    metric_names = [name.strip().upper() for name in args.metrics.split(",") if name]
    rubric_metrics = _resolve_rubric_metrics(metric_names)

    import pandas as pd

    data: dict[str, list[Any]] = {
        "prompt": [case.prompt for case in cases],
        "session_inputs": [
            types.evals.SessionInput(user_id=f"eval-user-{idx:04d}", state={})
            for idx, _ in enumerate(cases)
        ],
        "case_id": [case.case_id for case in cases],
        "category": [case.category for case in cases],
        "criticality": [case.criticality for case in cases],
    }
    if any(case.reference for case in cases):
        data["reference"] = [case.reference or "" for case in cases]

    dataset = pd.DataFrame(data)

    client = vertexai.Client(project=args.project_id, location=args.location)
    inferred_dataset = client.evals.run_inference(
        agent=args.agent_resource_name,
        src=dataset,
    )

    create_kwargs: dict[str, Any] = {
        "dataset": inferred_dataset,
        "metrics": rubric_metrics,
    }
    if args.dest:
        create_kwargs["dest"] = args.dest

    try:
        from f1_agent.agent import root_agent

        create_kwargs["agent_info"] = types.evals.AgentInfo.load_from_agent(
            root_agent, args.agent_resource_name
        )
    except Exception as exc:
        print(f"Warning: could not load agent_info automatically: {exc}")

    evaluation_run = client.evals.create_evaluation_run(**create_kwargs)
    run_name = _run_name(evaluation_run)
    resolved_run = evaluation_run

    if run_name:
        resolved_run = _wait_for_run(
            client=client,
            run_name=run_name,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        )

    summary_payload = _to_plain(getattr(resolved_run, "summary_metrics", None))
    summary_scores = _extract_summary_scores(summary_payload, metric_names)

    report = {
        "project_id": args.project_id,
        "location": args.location,
        "agent_resource_name": args.agent_resource_name,
        "dataset_file": args.dataset_file,
        "dataset_size": len(cases),
        "metrics_requested": metric_names,
        "evaluation_run_name": run_name or "",
        "evaluation_state": _run_state(resolved_run),
        "summary_metrics": summary_scores,
        "raw_summary_metrics": summary_payload,
        "release": _build_release_metadata(),
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
