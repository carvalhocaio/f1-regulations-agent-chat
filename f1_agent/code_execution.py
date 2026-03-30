"""Restricted analytical Code Execution via Agent Engine sandboxes.

This module intentionally does not execute arbitrary user/model code.
It exposes a small allowlist of analytical task templates and runs those
templates in Agent Engine Code Execution sandboxes when enabled.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

import vertexai
from google import genai
from google.api_core import exceptions as gcp_exceptions

from f1_agent.env_utils import env_bool, env_int
from f1_agent.resilience import CircuitBreakerOpenError, run_with_retry

logger = logging.getLogger(__name__)

_ENABLED_ENV = "F1_CODE_EXECUTION_ENABLED"
_PROJECT_ENV = "F1_RAG_PROJECT_ID"
_LOCATION_ENV = "F1_CODE_EXECUTION_LOCATION"
_AGENT_ENGINE_NAME_ENV = "F1_CODE_EXECUTION_AGENT_ENGINE_NAME"
_AGENT_ENGINE_ID_ENV = "GOOGLE_CLOUD_AGENT_ENGINE_ID"
_TTL_SECONDS_ENV = "F1_CODE_EXECUTION_SANDBOX_TTL_SECONDS"
_MAX_ROWS_ENV = "F1_CODE_EXECUTION_MAX_ROWS"
_MAX_VALUES_ENV = "F1_CODE_EXECUTION_MAX_VALUES"

_DEFAULT_LOCATION = "us-central1"
_DEFAULT_TTL_SECONDS = 3600
_DEFAULT_MAX_ROWS = 500
_DEFAULT_MAX_VALUES = 2000

_TASK_SUMMARY_STATS = "summary_stats"
_TASK_WHAT_IF_POINTS = "what_if_points"
_TASK_DISTRIBUTION = "distribution_bins"


@dataclass(frozen=True)
class CodeExecutionSettings:
    enabled: bool
    project_id: str
    location: str
    agent_engine_name: str
    sandbox_ttl_seconds: int
    max_rows: int
    max_values: int


def run_analytical_code(task_type: str, payload: str = "{}") -> dict:
    """Run an allowlisted analytical template in Code Execution sandbox.

    Args:
        task_type: One of summary_stats, what_if_points, distribution_bins.
        payload: JSON string with template-specific inputs.
    """
    settings = _load_settings()
    if not settings.enabled:
        return {
            "status": "disabled",
            "message": "Code Execution is disabled by configuration.",
        }

    if not settings.project_id:
        return {
            "status": "configuration_error",
            "message": f"Missing {_PROJECT_ENV} for Code Execution.",
        }

    if settings.location != "us-central1":
        return {
            "status": "configuration_error",
            "message": "Code Execution currently supports us-central1 only.",
        }

    if not settings.agent_engine_name:
        return {
            "status": "configuration_error",
            "message": (
                "Missing agent engine resource name for sandbox execution. "
                f"Set {_AGENT_ENGINE_NAME_ENV} or {_AGENT_ENGINE_ID_ENV}."
            ),
        }

    try:
        parsed_payload = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        return {"status": "error", "message": f"Invalid JSON payload: {exc}"}

    if not isinstance(parsed_payload, dict):
        return {
            "status": "error",
            "message": "Payload must be a JSON object.",
        }

    try:
        normalized_payload = _validate_payload(task_type, parsed_payload, settings)
        code = _build_code_template(task_type, normalized_payload)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    return _execute_in_sandbox(settings, task_type, normalized_payload, code)


def _load_settings() -> CodeExecutionSettings:
    project_id = os.environ.get(_PROJECT_ENV, "").strip()
    location = (
        os.environ.get(_LOCATION_ENV, _DEFAULT_LOCATION).strip() or _DEFAULT_LOCATION
    )
    configured_agent_name = os.environ.get(_AGENT_ENGINE_NAME_ENV, "").strip()
    fallback_agent = _agent_name_from_env(project_id, location)

    return CodeExecutionSettings(
        enabled=env_bool(_ENABLED_ENV, False),
        project_id=project_id,
        location=location,
        agent_engine_name=configured_agent_name or fallback_agent,
        sandbox_ttl_seconds=max(300, env_int(_TTL_SECONDS_ENV, _DEFAULT_TTL_SECONDS)),
        max_rows=max(10, env_int(_MAX_ROWS_ENV, _DEFAULT_MAX_ROWS)),
        max_values=max(100, env_int(_MAX_VALUES_ENV, _DEFAULT_MAX_VALUES)),
    )


def _agent_name_from_env(project_id: str, location: str) -> str:
    value = os.environ.get(_AGENT_ENGINE_ID_ENV, "").strip()
    if not value:
        return ""

    if value.startswith("projects/"):
        return value

    if not project_id:
        return ""

    return f"projects/{project_id}/locations/{location}/reasoningEngines/{value}"


def _validate_payload(
    task_type: str,
    payload: dict[str, Any],
    settings: CodeExecutionSettings,
) -> dict[str, Any]:
    if task_type == _TASK_SUMMARY_STATS:
        rows = payload.get("rows")
        field = str(payload.get("field", "")).strip()
        if not isinstance(rows, list) or not rows:
            raise ValueError("summary_stats requires non-empty list `rows`.")
        if len(rows) > settings.max_rows:
            raise ValueError(f"rows exceeds limit ({settings.max_rows}).")
        if not field:
            raise ValueError("summary_stats requires `field`.")
        return {"rows": rows, "field": field}

    if task_type == _TASK_WHAT_IF_POINTS:
        drivers = payload.get("drivers")
        if not isinstance(drivers, list) or not drivers:
            raise ValueError("what_if_points requires non-empty list `drivers`.")
        if len(drivers) > settings.max_rows:
            raise ValueError(f"drivers exceeds limit ({settings.max_rows}).")
        normalized = []
        for item in drivers:
            if not isinstance(item, dict):
                raise ValueError("Each entry in drivers must be an object.")
            name = str(item.get("name", "")).strip()
            current = _to_float(item.get("current_points", 0))
            delta = _to_float(item.get("delta_points", 0))
            if not name:
                raise ValueError("Driver entries require `name`.")
            normalized.append(
                {
                    "name": name,
                    "current_points": current,
                    "delta_points": delta,
                }
            )
        top_n = max(1, min(int(payload.get("top_n", 10)), len(normalized)))
        return {"drivers": normalized, "top_n": top_n}

    if task_type == _TASK_DISTRIBUTION:
        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError("distribution_bins requires non-empty list `values`.")
        if len(values) > settings.max_values:
            raise ValueError(f"values exceeds limit ({settings.max_values}).")
        bins = max(2, min(int(payload.get("bins", 10)), 50))
        numeric = [_to_float(v) for v in values]
        return {"values": numeric, "bins": bins}

    raise ValueError(
        "Unsupported task_type. Allowed: summary_stats, what_if_points, distribution_bins"
    )


def _build_code_template(task_type: str, payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=True)

    if task_type == _TASK_SUMMARY_STATS:
        return (
            "import json\n"
            "import statistics\n"
            f"payload = json.loads({json.dumps(payload_json)})\n"
            "field = payload['field']\n"
            "values = []\n"
            "for row in payload['rows']:\n"
            "    if isinstance(row, dict) and field in row:\n"
            "        try:\n"
            "            values.append(float(row[field]))\n"
            "        except Exception:\n"
            "            pass\n"
            "if not values:\n"
            "    print(json.dumps({'status': 'error', 'message': 'No numeric values found'}))\n"
            "else:\n"
            "    values_sorted = sorted(values)\n"
            "    out = {\n"
            "      'status': 'success',\n"
            "      'count': len(values),\n"
            "      'min': min(values),\n"
            "      'max': max(values),\n"
            "      'mean': statistics.fmean(values),\n"
            "      'median': statistics.median(values),\n"
            "      'p25': values_sorted[int((len(values_sorted)-1) * 0.25)],\n"
            "      'p75': values_sorted[int((len(values_sorted)-1) * 0.75)]\n"
            "    }\n"
            "    print(json.dumps(out))\n"
        )

    if task_type == _TASK_WHAT_IF_POINTS:
        return (
            "import json\n"
            f"payload = json.loads({json.dumps(payload_json)})\n"
            "rows = []\n"
            "for item in payload['drivers']:\n"
            "    projected = float(item['current_points']) + float(item.get('delta_points', 0))\n"
            "    rows.append({\n"
            "      'name': item['name'],\n"
            "      'current_points': float(item['current_points']),\n"
            "      'delta_points': float(item.get('delta_points', 0)),\n"
            "      'projected_points': projected\n"
            "    })\n"
            "rows.sort(key=lambda x: (-x['projected_points'], x['name']))\n"
            "top_n = int(payload.get('top_n', 10))\n"
            "for i, row in enumerate(rows, start=1):\n"
            "    row['projected_rank'] = i\n"
            "out = {'status': 'success', 'table': rows[:top_n]}\n"
            "print(json.dumps(out))\n"
        )

    if task_type == _TASK_DISTRIBUTION:
        return (
            "import json\n"
            f"payload = json.loads({json.dumps(payload_json)})\n"
            "values = [float(v) for v in payload['values']]\n"
            "bins = int(payload['bins'])\n"
            "vmin = min(values)\n"
            "vmax = max(values)\n"
            "if vmax == vmin:\n"
            "    edges = [vmin, vmax]\n"
            "    counts = [len(values)]\n"
            "else:\n"
            "    width = (vmax - vmin) / bins\n"
            "    edges = [vmin + i * width for i in range(bins + 1)]\n"
            "    counts = [0] * bins\n"
            "    for value in values:\n"
            "        idx = int((value - vmin) / width)\n"
            "        if idx >= bins:\n"
            "            idx = bins - 1\n"
            "        counts[idx] += 1\n"
            "out = {'status': 'success', 'min': vmin, 'max': vmax, 'bins': bins, 'edges': edges, 'counts': counts}\n"
            "print(json.dumps(out))\n"
        )

    raise ValueError("Unexpected task_type")


def _execute_in_sandbox(
    settings: CodeExecutionSettings,
    task_type: str,
    payload: dict[str, Any],
    code: str,
) -> dict[str, Any]:
    client = vertexai.Client(project=settings.project_id, location=settings.location)
    sandbox_name = None

    try:
        create_config = genai.types.CreateAgentEngineSandboxConfig(
            display_name=f"f1-analytics-{uuid.uuid4().hex[:8]}",
            ttl=f"{settings.sandbox_ttl_seconds}s",
        )
        operation = client.agent_engines.sandboxes.create(
            name=settings.agent_engine_name,
            spec={"code_execution_environment": {}},
            config=create_config,
        )
        sandbox_name = _extract_sandbox_name(operation)
        if not sandbox_name:
            return {
                "status": "error",
                "message": "Sandbox creation did not return a sandbox name.",
            }

        response = run_with_retry(
            "code_execution.sandbox_execute",
            lambda: client.agent_engines.sandboxes.execute_code(
                name=sandbox_name,
                input_data={"code": code},
            ),
            logger_instance=logger,
        )
        output = _extract_execution_output(response)

        result_payload = _try_parse_json_line(output.get("stdout", ""))

        return {
            "status": "success" if not output.get("stderr") else "partial_success",
            "task_type": task_type,
            "result": result_payload,
            "stdout": output.get("stdout", ""),
            "stderr": output.get("stderr", ""),
            "sandbox_name": sandbox_name,
            "input_preview": {
                "task_type": task_type,
                "keys": sorted(payload.keys()),
            },
        }
    except (
        gcp_exceptions.PermissionDenied,
        gcp_exceptions.InvalidArgument,
        gcp_exceptions.InternalServerError,
        gcp_exceptions.ServiceUnavailable,
        gcp_exceptions.DeadlineExceeded,
        CircuitBreakerOpenError,
    ) as exc:
        logger.warning("Code Execution sandbox failed", exc_info=True)
        return {
            "status": "error",
            "message": f"Code Execution failed: {exc}",
            "task_type": task_type,
        }
    finally:
        if sandbox_name:
            try:
                client.agent_engines.sandboxes.delete(name=sandbox_name)
            except Exception:
                logger.debug("Sandbox cleanup failed", exc_info=True)


def _extract_sandbox_name(operation: Any) -> str | None:
    response = getattr(operation, "response", None)
    if response is not None:
        name = getattr(response, "name", None)
        if name:
            return str(name)

    name = getattr(operation, "name", None)
    if name and "/sandboxes/" in str(name):
        return str(name)

    if isinstance(operation, dict):
        nested = operation.get("response")
        if isinstance(nested, dict):
            nested_name = nested.get("name")
            if nested_name:
                return str(nested_name)
        raw_name = operation.get("name")
        if raw_name and "/sandboxes/" in str(raw_name):
            return str(raw_name)

    return None


def _extract_execution_output(response: Any) -> dict[str, str]:
    stdout = _field(response, "stdout")
    stderr = _field(response, "stderr")

    if isinstance(response, dict):
        data = response.get("output_data")
        if isinstance(data, dict):
            stdout = stdout or data.get("stdout")
            stderr = stderr or data.get("stderr")

    output_data = _field(response, "output_data", "outputData")
    if output_data is not None:
        stdout = stdout or _field(output_data, "stdout")
        stderr = stderr or _field(output_data, "stderr")

    return {
        "stdout": str(stdout or ""),
        "stderr": str(stderr or ""),
    }


def _field(value: Any, *keys: str) -> Any:
    if value is None:
        return None

    for key in keys:
        if isinstance(value, dict) and key in value:
            return value[key]
        candidate = getattr(value, key, None)
        if candidate is not None:
            return candidate

    return None


def _try_parse_json_line(stdout: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    if not lines:
        return None

    for candidate in reversed(lines):
        if not (candidate.startswith("{") and candidate.endswith("}")):
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric value, got {value!r}") from exc
