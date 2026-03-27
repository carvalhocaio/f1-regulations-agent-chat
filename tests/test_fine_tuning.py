import json
import unittest

from f1_agent.fine_tuning.schema import (
    SYSTEM_INSTRUCTION,
    TOOL_DECLARATIONS,
    build_example,
    examples_to_jsonl,
)


class SchemaTests(unittest.TestCase):
    def test_build_example_basic(self):
        ex = build_example(
            user_message="Who won 2023?",
            model_answer="Verstappen won 2023.",
        )
        # Only "contents" at top level (matches Vertex AI SFT format)
        self.assertEqual(list(ex.keys()), ["contents"])
        self.assertNotIn("system_instruction", ex)
        self.assertNotIn("tools", ex)

        # Contents: user + model
        self.assertEqual(len(ex["contents"]), 2)
        self.assertEqual(ex["contents"][0]["role"], "user")
        self.assertEqual(ex["contents"][1]["role"], "model")

    def test_system_instruction_in_user_message(self):
        """System instruction is embedded in the first user message."""
        ex = build_example(
            user_message="Who won 2023?",
            model_answer="Verstappen won 2023.",
        )
        user_text = ex["contents"][0]["parts"][0]["text"]
        self.assertIn("[System:", user_text)
        self.assertIn("Formula 1", user_text)
        self.assertIn("Who won 2023?", user_text)

    def test_build_example_with_function_calls(self):
        """Function calls are converted to text in the model answer."""
        ex = build_example(
            user_message="Who won 2023?",
            function_calls=[
                {
                    "name": "query_f1_history_template",
                    "args": {
                        "template_name": "driver_champions",
                        "params": '{"year": 2023}',
                    },
                }
            ],
            function_responses=[
                {
                    "name": "query_f1_history_template",
                    "response": {
                        "status": "success",
                        "results": [{"driver": "Verstappen"}],
                    },
                }
            ],
            model_answer="Verstappen won 2023.",
        )

        # Text-only: user + model
        self.assertEqual(len(ex["contents"]), 2)

        # Model answer contains tool call text + original answer
        model_text = ex["contents"][1]["parts"][0]["text"]
        self.assertIn("[Tool Use]", model_text)
        self.assertIn("query_f1_history_template", model_text)
        self.assertIn("[Tool Results]", model_text)
        self.assertIn("Verstappen won 2023.", model_text)

    def test_build_example_custom_system_instruction(self):
        ex = build_example(
            user_message="Hello",
            model_answer="Hi!",
            system_instruction="Custom instruction",
        )
        user_text = ex["contents"][0]["parts"][0]["text"]
        self.assertIn("Custom instruction", user_text)

    def test_tool_declarations_have_required_fields(self):
        """TOOL_DECLARATIONS are still defined for reference/documentation."""
        for decl in TOOL_DECLARATIONS:
            self.assertIn("name", decl)
            self.assertIn("description", decl)
            self.assertIn("parameters", decl)
            self.assertIn("type", decl["parameters"])
            self.assertIn("properties", decl["parameters"])
            self.assertIn("additionalProperties", decl["parameters"])
            self.assertFalse(decl["parameters"]["additionalProperties"])

    def test_tool_declarations_names(self):
        names = [d["name"] for d in TOOL_DECLARATIONS]
        self.assertIn("query_f1_history_template", names)
        self.assertIn("query_f1_history", names)
        self.assertIn("search_regulations", names)
        self.assertIn("google_search", names)
        self.assertIn("run_analytical_code", names)

    def test_run_analytical_code_has_enum_task_type(self):
        declaration = next(
            d for d in TOOL_DECLARATIONS if d["name"] == "run_analytical_code"
        )
        task_type = declaration["parameters"]["properties"]["task_type"]
        self.assertEqual(
            task_type["enum"],
            ["summary_stats", "what_if_points", "distribution_bins"],
        )

    def test_examples_to_jsonl(self):
        examples = [
            build_example(user_message="Q1", model_answer="A1"),
            build_example(user_message="Q2", model_answer="A2"),
        ]
        jsonl = examples_to_jsonl(examples)
        lines = jsonl.strip().split("\n")
        self.assertEqual(len(lines), 2)

        for line in lines:
            parsed = json.loads(line)
            self.assertIn("contents", parsed)
            self.assertEqual(list(parsed.keys()), ["contents"])

    def test_system_instruction_not_empty(self):
        self.assertTrue(len(SYSTEM_INSTRUCTION) > 50)

    def test_no_function_call_modality(self):
        """Ensure no example contains functionCall/functionResponse parts."""
        ex = build_example(
            user_message="Q",
            function_calls=[{"name": "search_regulations", "args": {"query": "DRS"}}],
            function_responses=[
                {
                    "name": "search_regulations",
                    "response": {"status": "success", "results": []},
                }
            ],
            model_answer="A",
        )
        for content in ex["contents"]:
            for part in content["parts"]:
                self.assertNotIn("functionCall", part)
                self.assertNotIn("functionResponse", part)


class DatasetGenerationTests(unittest.TestCase):
    def test_generate_and_split(self):
        """Test that generate_all produces examples and split works."""
        from f1_agent.fine_tuning.generate_dataset import generate_all, split_dataset

        examples = generate_all(seed=42)
        self.assertGreater(len(examples), 0)

        train, test = split_dataset(examples, test_ratio=0.2)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(train) + len(test), len(examples))

    def test_all_examples_valid_json(self):
        """Every generated example must be serializable to valid JSON."""
        from f1_agent.fine_tuning.generate_dataset import generate_all

        examples = generate_all(seed=42)
        for i, ex in enumerate(examples):
            try:
                json.dumps(ex, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                self.fail(f"Example {i} is not JSON-serializable: {e}")

    def test_all_examples_have_required_structure(self):
        """Every example must have only 'contents' (Vertex AI SFT format)."""
        from f1_agent.fine_tuning.generate_dataset import generate_all

        examples = generate_all(seed=42)
        for i, ex in enumerate(examples):
            self.assertIn("contents", ex, f"Example {i} missing contents")
            self.assertEqual(
                list(ex.keys()),
                ["contents"],
                f"Example {i} has extra keys: {list(ex.keys())}",
            )
            self.assertGreaterEqual(
                len(ex["contents"]), 2, f"Example {i} has <2 contents"
            )

    def test_no_function_call_modality_in_dataset(self):
        """No generated example should contain functionCall/functionResponse."""
        from f1_agent.fine_tuning.generate_dataset import generate_all

        examples = generate_all(seed=42)
        for i, ex in enumerate(examples):
            for j, content in enumerate(ex["contents"]):
                for part in content["parts"]:
                    self.assertNotIn(
                        "functionCall",
                        part,
                        f"Example {i}, content {j} has functionCall",
                    )
                    self.assertNotIn(
                        "functionResponse",
                        part,
                        f"Example {i}, content {j} has functionResponse",
                    )


if __name__ == "__main__":
    unittest.main()
