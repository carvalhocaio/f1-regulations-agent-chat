"""
Vertex AI SFT dataset schema helpers.

Builds JSONL examples in the format expected by Gemini supervised fine-tuning,
including function declarations for the F1 agent's tools.
"""

from __future__ import annotations

import json
from typing import Any

# ── Tool declarations (mirrors what the ADK agent exposes) ──────────────

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "query_f1_history_template",
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
        },
    },
    {
        "name": "query_f1_history",
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
        },
    },
    {
        "name": "search_regulations",
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
        },
    },
    {
        "name": "google_search_agent",
        "description": "Search the web for current/live F1 information.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "request": {
                    "type": "STRING",
                    "description": "Web search request.",
                },
            },
            "required": ["request"],
        },
    },
]

SYSTEM_INSTRUCTION = (
    "You are an expert assistant on Formula 1, with deep knowledge of the "
    "FIA 2026 regulations and the sport in general. Use the available tools "
    "to answer questions accurately. Respond in the same language the user "
    "is using. Be enthusiastic but precise."
)


def _tool_calls_to_text(
    function_calls: list[dict[str, Any]],
    function_responses: list[dict[str, Any]] | None,
) -> str:
    """Convert structured function calls/responses into a text representation.

    This is needed because gemini-2.5-flash SFT does not support functionCall
    / functionResponse modalities. Instead, we teach the model the reasoning
    pattern via text.
    """
    lines: list[str] = []
    lines.append("[Tool Use]")
    for fc in function_calls:
        args_str = json.dumps(fc["args"], ensure_ascii=False)
        lines.append(f"- Called **{fc['name']}**({args_str})")

    if function_responses:
        lines.append("\n[Tool Results]")
        for fr in function_responses:
            resp_str = json.dumps(fr["response"], ensure_ascii=False)
            lines.append(f"- **{fr['name']}** returned: {resp_str}")

    return "\n".join(lines)


def build_example(
    *,
    user_message: str,
    function_calls: list[dict[str, Any]] | None = None,
    function_responses: list[dict[str, Any]] | None = None,
    model_answer: str,
    system_instruction: str | None = None,
) -> dict[str, Any]:
    """Build a single SFT training example in Vertex AI JSONL format.

    Function calls are converted to text representation since gemini-2.5-flash
    does not support functionCall/functionResponse modalities in SFT datasets.

    Args:
        user_message: The user's question.
        function_calls: Optional list of function calls the model should make.
            Each dict: {"name": "tool_name", "args": {...}}
        function_responses: Optional list of function responses.
            Each dict: {"name": "tool_name", "response": {...}}
        model_answer: The final model text answer.
        system_instruction: Override the default system instruction.

    Returns:
        A dict ready to be serialized as one line of JSONL.
    """
    contents: list[dict[str, Any]] = []

    # 1. User message — prepend system instruction as context since Vertex AI
    #    SFT JSONL only supports {"contents": [...]} at the top level.
    sys_text = system_instruction or SYSTEM_INSTRUCTION
    user_text = f"[System: {sys_text}]\n\n{user_message}"
    contents.append(
        {
            "role": "user",
            "parts": [{"text": user_text}],
        }
    )

    # 2. If there are function calls, convert to text and prepend to model answer
    if function_calls:
        tool_text = _tool_calls_to_text(function_calls, function_responses)
        full_answer = f"{tool_text}\n\n{model_answer}"
    else:
        full_answer = model_answer

    # 3. Final model answer (includes tool reasoning if applicable)
    contents.append(
        {
            "role": "model",
            "parts": [{"text": full_answer}],
        }
    )

    return {"contents": contents}


def examples_to_jsonl(examples: list[dict[str, Any]]) -> str:
    """Convert a list of examples to JSONL string."""
    return "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples)
