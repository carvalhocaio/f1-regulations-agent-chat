import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_deploy_module():
    module_path = Path(__file__).resolve().parents[1] / "deployment" / "deploy.py"
    spec = importlib.util.spec_from_file_location("deploy", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deploy = _load_deploy_module()


class ValidateRuntimeScalingTests(unittest.TestCase):
    def test_accepts_valid_scaling_values(self):
        args = SimpleNamespace(
            min_instances=2,
            max_instances=6,
            container_concurrency=18,
        )

        deploy._validate_runtime_scaling(args)

    def test_rejects_max_instances_lower_than_min(self):
        args = SimpleNamespace(
            min_instances=4,
            max_instances=2,
            container_concurrency=18,
        )

        with self.assertRaisesRegex(
            ValueError, "--max-instances must be >= --min-instances"
        ):
            deploy._validate_runtime_scaling(args)

    def test_rejects_container_concurrency_below_one(self):
        args = SimpleNamespace(
            min_instances=1,
            max_instances=4,
            container_concurrency=0,
        )

        with self.assertRaisesRegex(ValueError, "--container-concurrency must be >= 1"):
            deploy._validate_runtime_scaling(args)


class BuildAgentEngineConfigTests(unittest.TestCase):
    def test_includes_runtime_scaling_controls(self):
        args = SimpleNamespace(
            display_name="f1-agent",
            description="F1 Regulations & History Agent",
            service_account="agent@example.iam.gserviceaccount.com",
            staging_bucket="gs://bucket",
            min_instances=2,
            max_instances=6,
            container_concurrency=18,
        )

        config = deploy.build_agent_engine_config(args, env_vars={"A": "B"})
        payload = config.model_dump(exclude_none=True)

        self.assertEqual(payload.get("min_instances"), 2)
        self.assertEqual(payload.get("max_instances"), 6)
        self.assertEqual(payload.get("container_concurrency"), 18)

    def test_agentless_update_config_drops_agent_required_fields(self):
        args = SimpleNamespace(
            display_name="f1-agent",
            description="F1 Regulations & History Agent",
            service_account="agent@example.iam.gserviceaccount.com",
            staging_bucket="gs://bucket",
            min_instances=2,
            max_instances=6,
            container_concurrency=18,
        )
        config = deploy.build_agent_engine_config(
            args,
            env_vars={"GEMINI_API_KEY": "x", "F1_RAG_BACKEND": "auto"},
        )

        reduced = deploy._agentless_update_config(config)
        payload = reduced.model_dump(exclude_none=True)

        self.assertEqual(payload.get("display_name"), "f1-agent")
        self.assertNotIn("requirements", payload)
        self.assertNotIn("extra_packages", payload)
        self.assertNotIn("env_vars", payload)
        self.assertNotIn("min_instances", payload)
        self.assertNotIn("max_instances", payload)
        self.assertNotIn("container_concurrency", payload)


class DeployErrorClassifierTests(unittest.TestCase):
    def test_detects_invalid_agent_callable_error(self):
        exc = TypeError(
            "agent_engine has none of the following callable methods: query"
        )

        self.assertTrue(deploy._is_invalid_agent_callable_error(exc))

    def test_ignores_unrelated_errors(self):
        exc = ValueError("something else")

        self.assertFalse(deploy._is_invalid_agent_callable_error(exc))


if __name__ == "__main__":
    unittest.main()
