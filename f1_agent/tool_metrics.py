"""Cloud Monitoring export for tool validation errors."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_ENABLED_ENV = "F1_TOOL_METRICS_EXPORT_ENABLED"
_PROJECT_ENV = "F1_TOOL_METRICS_PROJECT_ID"
_METRIC_TYPE = "custom.googleapis.com/f1_agent/tool_validation_errors"

_metric_client: Any = None


def emit_tool_validation_error_metric(*, tool_name: str, error_code: str) -> None:
    """Emit one increment to Cloud Monitoring for tool validation errors.

    Fail-open by design: any exporter failure is logged and ignored.
    """
    if not _env_bool(_ENABLED_ENV, False):
        return

    project_id = _resolve_project_id()
    if not project_id:
        logger.debug("tool metrics export skipped: missing project id")
        return

    try:
        client = _get_metric_client()
        request = _build_create_time_series_request(
            project_id=project_id,
            tool_name=tool_name,
            error_code=error_code,
            value=1,
        )
        client.create_time_series(request=request)
    except Exception:
        logger.warning(
            "tool metrics export failed | tool=%s code=%s",
            tool_name,
            error_code,
            exc_info=True,
        )


def _build_create_time_series_request(
    *,
    project_id: str,
    tool_name: str,
    error_code: str,
    value: int,
    end_time_seconds: int | None = None,
) -> dict[str, Any]:
    end_seconds = end_time_seconds if end_time_seconds is not None else int(time.time())
    return {
        "name": f"projects/{project_id}",
        "time_series": [
            {
                "metric": {
                    "type": _METRIC_TYPE,
                    "labels": {
                        "tool_name": tool_name,
                        "error_code": error_code,
                    },
                },
                "resource": {
                    "type": "global",
                    "labels": {"project_id": project_id},
                },
                "points": [
                    {
                        "interval": {"end_time": {"seconds": end_seconds}},
                        "value": {"int64_value": int(value)},
                    }
                ],
            }
        ],
    }


def _get_metric_client() -> Any:
    global _metric_client
    if _metric_client is None:
        from google.cloud import monitoring_v3

        _metric_client = monitoring_v3.MetricServiceClient()
    return _metric_client


def _resolve_project_id() -> str:
    return (
        os.environ.get(_PROJECT_ENV, "").strip()
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
