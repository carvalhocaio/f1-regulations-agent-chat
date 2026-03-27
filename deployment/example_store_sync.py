"""Create/sync Example Store data from curated JSONL files.

Usage example:
  uv run python deployment/example_store_sync.py \
    --project-id <PROJECT_ID> \
    --location us-central1 \
    --display-name "f1-real-errors" \
    --dataset data/example_store/manual_examples.v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import vertexai
from google.api_core import exceptions as gcp_exceptions

_MAX_UPSERT_BATCH = 5
_MAX_RETRIES = 4
_PLACEHOLDER_TOKENS = {
    "<PROJECT_ID>",
    "<PROJECT_NUMBER>",
    "<EXAMPLE_STORE_ID>",
}


def _load_example_store_module():
    from vertexai.preview import example_stores

    return example_stores


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _normalize_example(row: dict[str, Any]) -> dict[str, Any]:
    if "stored_contents_example" in row:
        return {"stored_contents_example": row["stored_contents_example"]}

    if "contents_example" in row:
        return {
            "stored_contents_example": {
                "contents_example": row["contents_example"],
                "search_key_generation_method": {"last_entry": {}},
            }
        }

    user = str(row.get("user", "")).strip()
    expected = row.get("expected", [])
    if not user or not isinstance(expected, list):
        raise ValueError(
            "Each simplified row must include non-empty `user` and list `expected`"
        )

    expected_contents = []
    for item in expected:
        if not isinstance(item, dict):
            raise ValueError("Entries in `expected` must be objects")

        item_type = item.get("type")
        if item_type == "model_text":
            expected_contents.append(
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": str(item.get("text", "")).strip()}],
                    }
                }
            )
            continue

        if item_type == "function_call":
            expected_contents.append(
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": str(item.get("name", "")).strip(),
                                    "args": item.get("args", {}),
                                }
                            }
                        ]
                    }
                }
            )
            continue

        if item_type == "function_response":
            expected_contents.append(
                {
                    "content": {
                        "parts": [
                            {
                                "function_response": {
                                    "name": str(item.get("name", "")).strip(),
                                    "response": item.get("response", {}),
                                }
                            }
                        ]
                    }
                }
            )
            continue

        raise ValueError(
            "Unsupported expected.type. Use model_text, function_call, or function_response"
        )

    return {
        "stored_contents_example": {
            "contents_example": {
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "expected_contents": expected_contents,
            },
            "search_key_generation_method": {"last_entry": {}},
        }
    }


def _chunks(items: list[dict[str, Any]], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _with_retry(label: str, fn):
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn()
        except (
            gcp_exceptions.InternalServerError,
            gcp_exceptions.ServiceUnavailable,
            gcp_exceptions.DeadlineExceeded,
        ) as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            sleep_seconds = min(10, 2 * attempt)
            print(
                f"{label} failed with transient error ({exc.__class__.__name__}); "
                f"retrying in {sleep_seconds}s ({attempt}/{_MAX_RETRIES})"
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"{label} failed after {_MAX_RETRIES} attempts: {last_exc}")


def _looks_like_placeholder(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if any(token in text for token in _PLACEHOLDER_TOKENS):
        return True
    return "<" in text or ">" in text


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/sync Vertex Example Store")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--dataset", required=True, help="Path to curated JSONL file")
    parser.add_argument(
        "--example-store-name",
        default="",
        help="Existing Example Store resource name. If omitted, create a new store.",
    )
    parser.add_argument(
        "--display-name",
        default="f1-real-errors",
        help="Display name used only when creating a new store",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-multilingual-embedding-002",
        help="Embedding model for a new store",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if _looks_like_placeholder(args.project_id):
        raise ValueError(
            "--project-id contains placeholder text. Replace with the real project id, "
            "for example: f1-regulations-agent-chat"
        )

    if args.example_store_name and _looks_like_placeholder(args.example_store_name):
        raise ValueError(
            "--example-store-name contains placeholder text. Use the real resource "
            "name format: projects/<REAL_PROJECT_NUMBER>/locations/us-central1/exampleStores/<REAL_EXAMPLE_STORE_ID>"
        )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_rows = _read_jsonl(dataset_path)
    normalized_rows = [_normalize_example(row) for row in raw_rows]

    print(f"Loaded {len(normalized_rows)} curated examples from {dataset_path}")
    if not normalized_rows:
        print("No examples found; exiting")
        return

    if args.dry_run:
        print("Dry run enabled: no API calls were made")
        return

    vertexai.init(project=args.project_id, location=args.location)
    example_stores = _load_example_store_module()

    try:
        if args.example_store_name:
            example_store = _with_retry(
                "Load existing Example Store",
                lambda: example_stores.ExampleStore(args.example_store_name),
            )
        else:
            example_store = _with_retry(
                "Create Example Store",
                lambda: example_stores.ExampleStore.create(
                    display_name=args.display_name,
                    example_store_config=example_stores.ExampleStoreConfig(
                        vertex_embedding_model=args.embedding_model
                    ),
                ),
            )
    except gcp_exceptions.PermissionDenied as exc:
        message = str(exc)
        if "CONSUMER_INVALID" in message:
            raise RuntimeError(
                "Permission denied with CONSUMER_INVALID. Usually this means the "
                "resource name still has placeholders or the wrong project.\n"
                "Check:\n"
                "1) --project-id is a real project id\n"
                "2) --example-store-name uses real values (no <...>)\n"
                "3) API aiplatform.googleapis.com is enabled in that project\n"
                "4) Your active gcloud account has Vertex AI permissions"
            ) from exc
        raise

    store_name = getattr(example_store, "name", args.example_store_name)
    print(f"Using Example Store: {store_name}")

    uploaded = 0
    for batch in _chunks(normalized_rows, _MAX_UPSERT_BATCH):
        _with_retry(
            "Upsert examples batch", lambda b=batch: example_store.upsert_examples(b)
        )
        uploaded += len(batch)
        print(f"Upserted {uploaded}/{len(normalized_rows)} examples")

    print("Sync completed")


if __name__ == "__main__":
    main()
