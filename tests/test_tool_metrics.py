import os
import unittest
from unittest.mock import patch

from f1_agent import tool_metrics


class _FakeMetricClient:
    def __init__(self):
        self.requests = []

    def create_time_series(self, request):
        self.requests.append(request)


class ToolMetricsTests(unittest.TestCase):
    def tearDown(self):
        tool_metrics._metric_client = None

    def test_build_request_contains_expected_shape(self):
        request = tool_metrics._build_create_time_series_request(
            project_id="proj-123",
            tool_name="query_f1_history",
            error_code="INVALID_QUERY",
            value=1,
            end_time_seconds=1730000000,
        )

        self.assertEqual(request["name"], "projects/proj-123")
        series = request["time_series"][0]
        self.assertEqual(
            series["metric"]["type"],
            "custom.googleapis.com/f1_agent/tool_validation_errors",
        )
        self.assertEqual(series["metric"]["labels"]["tool_name"], "query_f1_history")
        self.assertEqual(series["metric"]["labels"]["error_code"], "INVALID_QUERY")
        self.assertEqual(series["resource"]["type"], "global")
        self.assertEqual(series["points"][0]["value"]["int64_value"], 1)

    def test_emit_skips_when_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("f1_agent.tool_metrics._get_metric_client") as mock_client:
                tool_metrics.emit_tool_validation_error_metric(
                    tool_name="search_regulations",
                    error_code="INVALID_ARGUMENT",
                )

        mock_client.assert_not_called()

    def test_emit_calls_cloud_monitoring_when_enabled(self):
        fake_client = _FakeMetricClient()
        with patch.dict(
            os.environ,
            {
                "F1_TOOL_METRICS_EXPORT_ENABLED": "true",
                "F1_TOOL_METRICS_PROJECT_ID": "proj-abc",
            },
            clear=True,
        ):
            with patch(
                "f1_agent.tool_metrics._get_metric_client", return_value=fake_client
            ):
                tool_metrics.emit_tool_validation_error_metric(
                    tool_name="query_f1_history_template",
                    error_code="INVALID_ARGUMENT",
                )

        self.assertEqual(len(fake_client.requests), 1)
        req = fake_client.requests[0]
        self.assertEqual(req["name"], "projects/proj-abc")


if __name__ == "__main__":
    unittest.main()
