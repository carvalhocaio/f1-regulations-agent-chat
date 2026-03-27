import os
from datetime import datetime
from pathlib import Path

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types

from f1_agent.callbacks import (
    check_cache,
    detect_corrections,
    inject_corrections,
    inject_dynamic_examples,
    inject_long_term_memories,
    inject_runtime_temporal_context,
    route_model,
    store_cache,
    sync_memory_bank,
)
from f1_agent.code_execution import run_analytical_code
from f1_agent.resilience import is_quota_or_unavailable_error
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
        YEAR_MINUS_1=CURRENT_YEAR - 1,
        YEAR_MINUS_2=CURRENT_YEAR - 2,
        YEAR_MINUS_3=CURRENT_YEAR - 3,
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

    if is_quota_or_unavailable_error(err) and (
        "429" in err_text
        or "ResourceExhausted" in err_text
        or "resourceexhausted" in err_text.lower()
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

    if is_quota_or_unavailable_error(err) and (
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def _build_model():
    if not _env_bool("F1_LLM_RETRY_ENABLED", True):
        return "gemini-2.5-pro"

    retry_options = types.HttpRetryOptions(
        initial_delay=max(0.0, _env_float("F1_LLM_RETRY_INITIAL_DELAY_S", 1.0)),
        attempts=max(1, _env_int("F1_LLM_RETRY_ATTEMPTS", 3)),
        exp_base=max(1.1, _env_float("F1_LLM_RETRY_EXP_BASE", 2.0)),
        max_delay=max(0.1, _env_float("F1_LLM_RETRY_MAX_DELAY_S", 8.0)),
        jitter=max(0.0, _env_float("F1_LLM_RETRY_JITTER", 0.35)),
        http_status_codes=[408, 429, 500, 502, 503, 504],
    )
    return Gemini(model="gemini-2.5-pro", retry_options=retry_options)


root_agent = Agent(
    name="f1_regulations_assistant",
    model=_build_model(),
    description=(
        "An AI assistant for Formula 1, covering both the official FIA 2026 "
        "regulations and general F1 knowledge."
    ),
    instruction=_load_instruction(),
    tools=[
        search_regulations,
        query_f1_history_template,
        query_f1_history,
        run_analytical_code,
        search,
        google_search,
    ],
    before_model_callback=[
        check_cache,
        inject_runtime_temporal_context,
        inject_corrections,
        inject_long_term_memories,
        inject_dynamic_examples,
        route_model,
    ],
    after_model_callback=[detect_corrections, sync_memory_bank, store_cache],
    on_model_error_callback=handle_rate_limit,
)
