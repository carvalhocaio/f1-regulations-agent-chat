# Live fine-tuning dataset (anonymized)

This directory stores versioned datasets for the continuous SFT loop (Q5):

- `live_failures.<version>.jsonl`: curated and redacted failure cases.
- `dataset.train.<version>.jsonl`: Vertex SFT training JSONL.
- `dataset.test.<version>.jsonl`: Vertex SFT validation/test JSONL.
- `manifest.<version>.json`: generation metadata and summary stats.

## Privacy and governance default

By default, the builder applies redaction before writing files:

- emails -> `<REDACTED_EMAIL>`
- URLs -> `<REDACTED_URL>`
- phone numbers -> `<REDACTED_PHONE>`
- long numeric IDs -> `<REDACTED_ID>`
- key/value secrets (`api_key=...`, `token: ...`) -> `<REDACTED_SECRET>`

Use `--disable-redaction` only for controlled, short-lived debugging datasets.

## Input schema (raw failures)

Each line in the source failures JSONL should include:

- `prompt` (required)
- `expected_behavior` (required)
- `observed_response` (optional)
- `category` (optional, default `general`)
- `criticality` (optional: `high|medium|low`, default `medium`)
- `failure_type` (optional, default `unknown`)
- `source` (optional, default `runtime`)
- `timestamp_utc` (optional)
- `metadata` (optional object)

## Build command

```bash
uv run python deployment/collect_live_failures.py \
  --eval-dataset-file data/evals/agent_regression.v1.jsonl \
  --eval-report-file eval_report.json \
  --eval-gate-result-file eval_gate_result.json \
  --output-file data/fine_tuning_live/live_failures.raw.v1.jsonl

uv run python deployment/build_live_fine_tuning_dataset.py \
  --failures-file data/fine_tuning_live/live_failures.raw.v1.jsonl \
  --output-dir data/fine_tuning_live \
  --version v1
```

Then upload the generated train/test files to GCS and run tuning as usual.
