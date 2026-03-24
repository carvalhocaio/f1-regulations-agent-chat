from datetime import datetime
from pathlib import Path

from google.adk.agents import Agent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types

from f1_agent.callbacks import check_cache, route_model, store_cache
from f1_agent.tools import (
    query_f1_history,
    query_f1_history_template,
    search,
    search_regulations,
)

CURRENT_YEAR = datetime.now().year

google_search = GoogleSearchTool(bypass_multi_tools_limit=True)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_instruction() -> str:
    """Load the system instruction from file and interpolate runtime values."""
    template = (_PROMPTS_DIR / "system_instruction.txt").read_text(encoding="utf-8")
    return template.format(
        CURRENT_YEAR=CURRENT_YEAR,
        YEAR_MINUS_4=CURRENT_YEAR - 4,
    )


def handle_rate_limit(
    callback_context,
    llm_request,
    error=None,
    exception=None,
    **_kwargs,
):
    err = error or exception
    if err is None:
        return None

    err_text = str(err)
    err_type = type(err).__name__

    if (
        "429" in err_text
        or "ResourceExhausted" in err_type
        or "ResourceExhausted" in err_text
    ):
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            "⏳ Oops! The per-minute request limit has"
                            " been reached. Please wait a moment and try"
                            " again!"
                        )
                    )
                ],
            )
        )

    if (
        "503" in err_text
        or "Service Unavailable" in err_text
        or "UNAVAILABLE" in err_text
    ):
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            "⏳ The model is under high demand right now (503). "
                            "Please try again in a few moments."
                        )
                    )
                ],
            )
        )

    return None


root_agent = Agent(
    name="f1_regulations_assistant",
    model="gemini-2.5-pro",
    description=(
        "An AI assistant for Formula 1, covering both the official FIA 2026 "
        "regulations and general F1 knowledge."
    ),
    instruction=_load_instruction(),
    tools=[
        search_regulations,
        query_f1_history_template,
        query_f1_history,
        search,
        google_search,
    ],
    before_model_callback=[check_cache, route_model],
    after_model_callback=store_cache,
    on_model_error_callback=handle_rate_limit,
)
