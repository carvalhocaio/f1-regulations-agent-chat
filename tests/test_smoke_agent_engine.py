import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_smoke_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "deployment" / "smoke_agent_engine.py"
    )
    spec = importlib.util.spec_from_file_location("smoke_agent_engine", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke_module()


class ResourceNameTests(unittest.TestCase):
    def test_reads_resource_name_attribute(self):
        engine = SimpleNamespace(
            resource_name="projects/p/locations/l/reasoningEngines/r"
        )
        self.assertEqual(
            smoke._resource_name(engine), "projects/p/locations/l/reasoningEngines/r"
        )

    def test_reads_name_from_api_resource_dict(self):
        engine = SimpleNamespace(
            api_resource={"name": "projects/p/locations/l/reasoningEngines/r"}
        )
        self.assertEqual(
            smoke._resource_name(engine), "projects/p/locations/l/reasoningEngines/r"
        )


class DisplayNameTests(unittest.TestCase):
    def test_reads_display_name_attribute(self):
        engine = SimpleNamespace(display_name="f1-agent")
        self.assertEqual(smoke._display_name(engine), "f1-agent")

    def test_reads_display_name_from_api_resource_dict(self):
        engine = SimpleNamespace(api_resource={"displayName": "f1-agent"})
        self.assertEqual(smoke._display_name(engine), "f1-agent")


if __name__ == "__main__":
    unittest.main()
