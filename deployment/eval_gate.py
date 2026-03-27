"""Apply quality gates to an evaluation report."""

from __future__ import annotations

import argparse
import json
from numbers import Number
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _metric_value(metrics: dict[str, Any], metric_name: str) -> float | None:
    value = metrics.get(metric_name)
    if isinstance(value, Number):
        return float(value)
    return None


def evaluate_gate(
    *,
    report: dict[str, Any],
    thresholds: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    current_metrics = report.get("summary_metrics")
    if not isinstance(current_metrics, dict):
        raise ValueError("report.summary_metrics must be a JSON object")

    baseline_metrics: dict[str, Any] = {}
    if baseline:
        baseline_payload = baseline.get("summary_metrics")
        if isinstance(baseline_payload, dict):
            baseline_metrics = baseline_payload

    metric_thresholds = thresholds.get("metrics")
    if not isinstance(metric_thresholds, dict) or not metric_thresholds:
        raise ValueError("thresholds.metrics must be a non-empty object")

    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    for metric_name, config in metric_thresholds.items():
        if not isinstance(config, dict):
            raise ValueError(f"threshold config for {metric_name} must be an object")

        absolute_min = config.get("absolute_min")
        max_regression_delta = config.get("max_regression_delta")
        current_value = _metric_value(current_metrics, metric_name)
        baseline_value = _metric_value(baseline_metrics, metric_name)

        check = {
            "metric": metric_name,
            "current": current_value,
            "baseline": baseline_value,
            "absolute_min": absolute_min,
            "max_regression_delta": max_regression_delta,
            "passed": True,
            "notes": [],
        }

        if current_value is None:
            check["passed"] = False
            check["notes"].append("missing current metric")
            failures.append(f"{metric_name}: missing current metric")
            checks.append(check)
            continue

        if isinstance(absolute_min, Number) and current_value < float(absolute_min):
            check["passed"] = False
            check["notes"].append("below absolute minimum")
            failures.append(
                f"{metric_name}: {current_value:.4f} < absolute_min {float(absolute_min):.4f}"
            )

        if baseline_value is None:
            check["notes"].append("baseline missing; regression check skipped")
        elif isinstance(max_regression_delta, Number):
            drop = baseline_value - current_value
            if drop > float(max_regression_delta):
                check["passed"] = False
                check["notes"].append("regression delta exceeded")
                failures.append(
                    f"{metric_name}: regression {drop:.4f} > max_regression_delta {float(max_regression_delta):.4f}"
                )

        checks.append(check)

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "checks": checks,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce eval quality thresholds and regression deltas"
    )
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--thresholds-file", required=True)
    parser.add_argument("--baseline-file", default="")
    parser.add_argument("--output-file", default="eval_gate_result.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    report = _load_json(Path(args.report_file))
    thresholds = _load_json(Path(args.thresholds_file))
    baseline = None
    if args.baseline_file:
        baseline_path = Path(args.baseline_file)
        if baseline_path.exists():
            baseline = _load_json(baseline_path)

    result = evaluate_gate(report=report, thresholds=thresholds, baseline=baseline)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
