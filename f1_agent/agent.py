import os
import warnings
from datetime import datetime
from pathlib import Path

from google.genai import types

from f1_agent.adk_compat import (
    Agent,
    App,
    Gemini,
    GoogleSearchTool,
    LlmResponse,
)
from f1_agent.callbacks import (
    apply_grounding_policy,
    apply_response_contract,
    apply_throughput_request_type,
    check_cache,
    detect_corrections,
    inject_corrections,
    inject_dynamic_examples,
    inject_runtime_temporal_context,
    log_context_cache_metrics,
    preflight_token_check,
    route_model,
    store_cache,
    validate_grounding_outcome,
    validate_structured_response,
)
from f1_agent.code_execution import run_analytical_code
from f1_agent.env_utils import env_bool, env_float, env_int
from f1_agent.resilience import is_quota_or_unavailable_error
from f1_agent.tools import (
    get_current_season_info,
    query_f1_history,
    query_f1_history_template,
    search_recent_results,
    search_regulations,
)

# Suppress ADK experimental feature warnings
warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")

CURRENT_YEAR = datetime.now().year

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _build_tools():
    """Build the agent's tool list, conditionally including GoogleSearchTool."""
    tools = [
        search_regulations,
        query_f1_history_template,
        query_f1_history,
        run_analytical_code,
        get_current_season_info,
        search_recent_results,
    ]
    if env_bool("F1_GOOGLE_SEARCH_ENABLED", True):
        tools.append(GoogleSearchTool(bypass_multi_tools_limit=True))
    return tools


def _load_static_instruction() -> str:
    """Load the static system instruction from file and interpolate runtime values.

    This instruction is placed in ``static_instruction`` so that ADK keeps it
    as a stable system-instruction prefix, enabling Gemini context caching
    (both implicit and explicit).  Dynamic per-request content (temporal
    context, corrections, memories, examples) is injected separately into
    user content by before-model callbacks.
    """
    template = (_PROMPTS_DIR / "system_instruction_static.txt").read_text(
        encoding="utf-8"
    )
    return template.format(
        CURRENT_YEAR=CURRENT_YEAR,
        YEAR_MINUS_1=CURRENT_YEAR - 1,
        YEAR_MINUS_2=CURRENT_YEAR - 2,
        YEAR_MINUS_3=CURRENT_YEAR - 3,
        YEAR_MINUS_4=CURRENT_YEAR - 4,
    )


_PRO_QUOTA_STATE_KEY = "f1_pro_quota_exhausted_at"


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
        # Detect Pro-specific quota exhaustion and flag for Flash fallback
        is_pro_error = "gemini-2.5-pro" in err_text or (
            hasattr(llm_request, "model")
            and isinstance(llm_request.model, str)
            and "pro" in llm_request.model
        )
        state = getattr(callback_context, "state", None)
        if is_pro_error and isinstance(state, dict):
            state[_PRO_QUOTA_STATE_KEY] = datetime.now().isoformat()
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                "⚡ A cota do modelo Pro foi atingida."
                                " Alternando para o modelo Flash."
                                " Por favor, envie sua pergunta novamente!"
                            )
                        )
                    ],
                )
            )

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


_DEFAULT_MODEL = os.environ.get("F1_DEFAULT_MODEL", "").strip() or "gemini-2.5-pro"


def _build_model():
    if not env_bool("F1_LLM_RETRY_ENABLED", True):
        return _DEFAULT_MODEL

    retry_options = types.HttpRetryOptions(
        initial_delay=max(0.0, env_float("F1_LLM_RETRY_INITIAL_DELAY_S", 1.0)),
        attempts=max(1, env_int("F1_LLM_RETRY_ATTEMPTS", 3)),
        exp_base=max(1.1, env_float("F1_LLM_RETRY_EXP_BASE", 2.0)),
        max_delay=max(0.1, env_float("F1_LLM_RETRY_MAX_DELAY_S", 8.0)),
        jitter=max(0.0, env_float("F1_LLM_RETRY_JITTER", 0.35)),
        http_status_codes=[408, 429, 500, 502, 503, 504],
    )
    return Gemini(model=_DEFAULT_MODEL, retry_options=retry_options)


root_agent = Agent(
    name="f1_regulations_assistant",
    model=_build_model(),
    description=(
        "An AI assistant for Formula 1, covering both the official FIA 2026 "
        "regulations and general F1 knowledge."
    ),
    static_instruction=_load_static_instruction(),
    tools=_build_tools(),
    before_model_callback=[
        check_cache,
        inject_runtime_temporal_context,
        inject_corrections,
        inject_dynamic_examples,
        route_model,
        apply_throughput_request_type,
        apply_grounding_policy,
        apply_response_contract,
        preflight_token_check,
    ],
    after_model_callback=[
        log_context_cache_metrics,
        validate_structured_response,
        validate_grounding_outcome,
        detect_corrections,
        store_cache,
    ],
    on_model_error_callback=handle_rate_limit,
)


def build_app() -> App:
    """Wrap the root agent in an App container."""
    return App(
        name="f1_regulations_assistant",
        root_agent=root_agent,
    )
