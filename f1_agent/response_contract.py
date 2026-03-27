"""Structured response contracts for critical JSON payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CONTRACT_ID_SOURCES_BLOCK_V1 = "sources_block_v1"
CONTRACT_ID_COMPARISON_TABLE_V1 = "comparison_table_v1"


@dataclass(frozen=True)
class ResponseContract:
    contract_id: str
    response_mime_type: str
    response_schema: dict[str, Any]


_CONTRACTS: dict[str, ResponseContract] = {
    CONTRACT_ID_SOURCES_BLOCK_V1: ResponseContract(
        contract_id=CONTRACT_ID_SOURCES_BLOCK_V1,
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "required": ["answer", "sources", "schema_version"],
            "properties": {
                "schema_version": {"type": "STRING", "enum": ["v1"]},
                "answer": {"type": "STRING"},
                "sources": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "required": ["source_type", "title", "reference", "excerpt"],
                        "properties": {
                            "source_type": {
                                "type": "STRING",
                                "enum": ["regulation", "historical_db", "web"],
                            },
                            "title": {"type": "STRING"},
                            "reference": {"type": "STRING"},
                            "excerpt": {"type": "STRING"},
                            "url": {"type": "STRING"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    ),
    CONTRACT_ID_COMPARISON_TABLE_V1: ResponseContract(
        contract_id=CONTRACT_ID_COMPARISON_TABLE_V1,
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "required": ["schema_version", "title", "columns", "rows"],
            "properties": {
                "schema_version": {"type": "STRING", "enum": ["v1"]},
                "title": {"type": "STRING"},
                "columns": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
                "rows": {
                    "type": "ARRAY",
                    "items": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "notes": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
            },
            "additionalProperties": False,
        },
    ),
}


def get_response_contract(contract_id: str) -> ResponseContract | None:
    """Return response contract by id, or None when unknown."""
    return _CONTRACTS.get(contract_id)


def list_response_contract_ids() -> list[str]:
    """Return sorted list of supported response contract ids."""
    return sorted(_CONTRACTS.keys())


def validate_contract_payload(
    contract_id: str, payload: Any
) -> tuple[bool, str | None]:
    """Validate a parsed JSON payload against a supported response contract."""
    if contract_id == CONTRACT_ID_SOURCES_BLOCK_V1:
        return _validate_sources_block_v1(payload)
    if contract_id == CONTRACT_ID_COMPARISON_TABLE_V1:
        return _validate_comparison_table_v1(payload)
    return False, f"unsupported contract_id: {contract_id}"


def _validate_sources_block_v1(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "payload must be an object"

    schema_version = payload.get("schema_version")
    answer = payload.get("answer")
    sources = payload.get("sources")

    if schema_version != "v1":
        return False, "schema_version must be 'v1'"
    if not isinstance(answer, str):
        return False, "answer must be a string"
    if not isinstance(sources, list):
        return False, "sources must be an array"

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            return False, f"sources[{index}] must be an object"

        source_type = source.get("source_type")
        if source_type not in {"regulation", "historical_db", "web"}:
            return False, f"sources[{index}].source_type is invalid"

        for field in ("title", "reference", "excerpt"):
            if not isinstance(source.get(field), str):
                return False, f"sources[{index}].{field} must be a string"

        if "url" in source and not isinstance(source.get("url"), str):
            return False, f"sources[{index}].url must be a string"

    return True, None


def _validate_comparison_table_v1(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "payload must be an object"

    if payload.get("schema_version") != "v1":
        return False, "schema_version must be 'v1'"
    if not isinstance(payload.get("title"), str):
        return False, "title must be a string"

    columns = payload.get("columns")
    rows = payload.get("rows")
    notes = payload.get("notes")

    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        return False, "columns must be an array of strings"
    if not isinstance(rows, list):
        return False, "rows must be an array"

    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            return False, f"rows[{row_index}] must be an array"
        if not all(isinstance(cell, str) for cell in row):
            return False, f"rows[{row_index}] must contain only strings"

    if notes is not None:
        if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
            return False, "notes must be an array of strings"

    return True, None
