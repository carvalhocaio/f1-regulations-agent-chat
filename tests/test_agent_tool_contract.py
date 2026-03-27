import unittest

from f1_agent.agent import root_agent


class AgentToolContractTests(unittest.TestCase):
    def test_instruction_uses_google_search_agent_name(self):
        instruction = root_agent.static_instruction

        self.assertIn("google_search_agent", instruction)
        self.assertNotIn("**google_search**", instruction)

    def test_instruction_mentions_search_as_fallback(self):
        instruction = root_agent.static_instruction

        self.assertIn("search", instruction)
        self.assertIn("fallback", instruction.lower())

    def test_instruction_mentions_analytical_sandbox_tool(self):
        instruction = root_agent.static_instruction

        self.assertIn("run_analytical_code", instruction)
        self.assertIn("summary_stats", instruction)

    def test_instruction_loaded_from_template(self):
        instruction = root_agent.static_instruction

        # Should contain interpolated current year (not the placeholder)
        self.assertNotIn("{CURRENT_YEAR}", instruction)
        # Should contain few-shot examples
        self.assertIn("Example 1", instruction)
        self.assertIn("query_f1_history_template", instruction)

    def test_agent_registers_expected_tools(self):
        tool_names = []
        for tool in root_agent.tools:
            if hasattr(tool, "__name__"):
                tool_names.append(tool.__name__)
            elif hasattr(tool, "name"):
                tool_names.append(tool.name)

        self.assertIn("search_regulations", tool_names)
        self.assertIn("query_f1_history", tool_names)
        self.assertIn("query_f1_history_template", tool_names)
        self.assertIn("run_analytical_code", tool_names)
        self.assertIn("search", tool_names)
        self.assertIn("google_search", tool_names)

    def test_instruction_enforces_last_event_without_year_policy(self):
        instruction = root_agent.static_instruction

        self.assertIn("last completed edition", instruction)
        self.assertIn("DATE + YEAR", instruction)

    def test_instruction_enforces_preseason_leader_guard(self):
        instruction = root_agent.static_instruction

        self.assertIn("A temporada atual", instruction)
        self.assertIn("ainda não começou", instruction)

    def test_before_model_callback_includes_dynamic_examples(self):
        callback_names = [cb.__name__ for cb in root_agent.before_model_callback]

        self.assertIn("inject_long_term_memories", callback_names)
        self.assertIn("inject_dynamic_examples", callback_names)
        self.assertLess(
            callback_names.index("inject_corrections"),
            callback_names.index("inject_long_term_memories"),
        )
        self.assertLess(
            callback_names.index("inject_long_term_memories"),
            callback_names.index("inject_dynamic_examples"),
        )
        self.assertLess(
            callback_names.index("inject_dynamic_examples"),
            callback_names.index("route_model"),
        )

    def test_after_model_callback_includes_memory_sync(self):
        callback_names = [cb.__name__ for cb in root_agent.after_model_callback]

        self.assertIn("sync_memory_bank", callback_names)
        self.assertLess(
            callback_names.index("detect_corrections"),
            callback_names.index("sync_memory_bank"),
        )


if __name__ == "__main__":
    unittest.main()
