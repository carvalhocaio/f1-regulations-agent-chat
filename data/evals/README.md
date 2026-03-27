# Agent evaluation datasets

This directory contains versioned regression datasets used by the release quality gate.

## File format

Each row is JSONL with the following fields:

- `id` (string, required): stable case identifier.
- `category` (string, required): case bucket (e.g. `tool_routing`, `factuality`).
- `criticality` (string, required): `high`, `medium`, or `low`.
- `prompt` (string, required): input sent to the deployed agent.
- `reference` (string, optional): expected fact or guidance when available.

## Versioning

- Use semantic-style suffixes in file names (e.g. `agent_regression.v1.jsonl`, `agent_regression.v1.1.jsonl`).
- Never mutate historical files in place after release approval; add a new version.
- Keep changes reviewable: add/remove cases in small PRs and explain why in the PR description.

## Gate policy

The deploy workflow evaluates the candidate agent against this dataset and applies thresholds from `config/eval_thresholds.json`.
