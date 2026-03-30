"""Canonical function-calling contract for F1 agent tools."""

from __future__ import annotations

from typing import Any

TOOL_NAME_SEARCH_REGULATIONS = "search_regulations"
TOOL_NAME_QUERY_F1_HISTORY_TEMPLATE = "query_f1_history_template"
TOOL_NAME_QUERY_F1_HISTORY = "query_f1_history"
TOOL_NAME_RUN_ANALYTICAL_CODE = "run_analytical_code"
TOOL_NAME_GET_CURRENT_SEASON_INFO = "get_current_season_info"
TOOL_NAME_SEARCH_RECENT_RESULTS = "search_recent_results"
TOOL_NAME_GOOGLE_SEARCH = "google_search"

ALLOWED_TOOL_NAMES = [
    TOOL_NAME_SEARCH_REGULATIONS,
    TOOL_NAME_QUERY_F1_HISTORY_TEMPLATE,
    TOOL_NAME_QUERY_F1_HISTORY,
    TOOL_NAME_RUN_ANALYTICAL_CODE,
    TOOL_NAME_GET_CURRENT_SEASON_INFO,
    TOOL_NAME_SEARCH_RECENT_RESULTS,
    TOOL_NAME_GOOGLE_SEARCH,
]


def build_tool_declarations() -> list[dict[str, Any]]:
    """Return strict JSON-schema declarations for all runtime tools."""
    return [
        {
            "name": TOOL_NAME_QUERY_F1_HISTORY_TEMPLATE,
            "description": (
                "Execute a pre-built SQL template against the F1 historical "
                "database (1950-2024)."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "template_name": {
                        "type": "STRING",
                        "description": "Name of the SQL template.",
                    },
                    "params": {
                        "type": "STRING",
                        "description": "JSON string with template parameters.",
                    },
                },
                "required": ["template_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": TOOL_NAME_QUERY_F1_HISTORY,
            "description": (
                "Execute a raw SQLite SELECT query against the F1 historical "
                "database (1950-2024)."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "sql_query": {
                        "type": "STRING",
                        "description": "A SQLite SELECT query.",
                    },
                },
                "required": ["sql_query"],
                "additionalProperties": False,
            },
        },
        {
            "name": TOOL_NAME_SEARCH_REGULATIONS,
            "description": "Search the FIA 2026 F1 Regulations.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "Search query about F1 regulations.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": TOOL_NAME_RUN_ANALYTICAL_CODE,
            "description": (
                "Run restricted analytical templates in a managed sandbox for "
                "advanced computations."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "task_type": {
                        "type": "STRING",
                        "description": (
                            "Allowlisted analytical task: summary_stats, "
                            "what_if_points, distribution_bins."
                        ),
                        "enum": [
                            "summary_stats",
                            "what_if_points",
                            "distribution_bins",
                        ],
                    },
                    "payload": {
                        "type": "STRING",
                        "description": "JSON string with task-specific input data.",
                    },
                },
                "required": ["task_type"],
                "additionalProperties": False,
            },
        },
        {
            "name": TOOL_NAME_GET_CURRENT_SEASON_INFO,
            "description": (
                "Get the current F1 season calendar and identify which races "
                "have already happened. No parameters required."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": TOOL_NAME_SEARCH_RECENT_RESULTS,
            "description": (
                "Search for official F1 race results from the Jolpica API "
                "(2025 onwards). Returns accurate, structured race "
                "classifications. Prefer over google_search for results."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "year": {
                        "type": "INTEGER",
                        "description": "The season year (2025 or later).",
                    },
                    "race_round": {
                        "type": "INTEGER",
                        "description": (
                            "The race round number. If 0 or omitted, "
                            "returns the latest race with results."
                        ),
                    },
                },
                "required": ["year"],
                "additionalProperties": False,
            },
        },
        # Note: google_search is a Gemini built-in tool, not a function tool.
        # It does not need a declaration here.
    ]
