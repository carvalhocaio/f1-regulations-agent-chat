"""Build a versioned live fine-tuning dataset from curated failure records.

Default behavior redacts common sensitive patterns (emails, URLs, phones,
secrets, long IDs) before persisting files.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from f1_agent.fine_tuning.live_dataset import (
    build_sft_examples_from_failures,
    load_jsonl,
    normalize_failure_cases,
    split_examples,
    summarize_cases,
    write_jsonl,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build versioned train/test JSONL from live failure cases"
    )
    parser.add_argument("--failures-file", required=True)
    parser.add_argument("--output-dir", default="data/fine_tuning_live")
    parser.add_argument(
        "--version", required=True, help="Dataset version label (e.g. v1)"
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--disable-redaction",
        action="store_true",
        help="Disable redaction (not recommended)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = Path(args.failures_file)
    output_dir = Path(args.output_dir)

    raw_cases = load_jsonl(source_path)
    curated_cases = normalize_failure_cases(
        raw_cases,
        redact=not args.disable_redaction,
    )

    if args.max_examples > 0:
        curated_cases = curated_cases[: args.max_examples]

    if not curated_cases:
        raise SystemExit("No valid curated failure cases to export")

    sft_examples = build_sft_examples_from_failures(curated_cases)
    train_examples, test_examples = split_examples(
        sft_examples,
        test_ratio=max(0.0, min(0.5, args.test_ratio)),
        seed=args.seed,
    )

    live_cases_path = output_dir / f"live_failures.{args.version}.jsonl"
    train_path = output_dir / f"dataset.train.{args.version}.jsonl"
    test_path = output_dir / f"dataset.test.{args.version}.jsonl"
    manifest_path = output_dir / f"manifest.{args.version}.json"

    write_jsonl(live_cases_path, curated_cases)
    write_jsonl(train_path, train_examples)
    write_jsonl(test_path, test_examples)

    manifest = {
        "version": args.version,
        "source_file": str(source_path),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "redaction_enabled": not args.disable_redaction,
        "split": {
            "test_ratio": max(0.0, min(0.5, args.test_ratio)),
            "seed": args.seed,
        },
        "counts": {
            "curated_cases": len(curated_cases),
            "train_examples": len(train_examples),
            "test_examples": len(test_examples),
        },
        "summary": summarize_cases(curated_cases),
        "outputs": {
            "live_cases": str(live_cases_path),
            "train": str(train_path),
            "test": str(test_path),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
