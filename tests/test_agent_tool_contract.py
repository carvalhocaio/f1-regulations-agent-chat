import unittest

from f1_agent.agent import root_agent


class AgentToolContractTests(unittest.TestCase):
    def test_instruction_references_google_search(self):
        instruction = root_agent.static_instruction

        self.assertIn("google_search", instruction)
        self.assertNotIn("google_search_agent", instruction)

    def test_instruction_does_not_allow_search_alias(self):
        instruction = root_agent.static_instruction

        self.assertNotIn("`search`", instruction)
        self.assertIn("Never invent tool names", instruction)

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
        self.assertIn("get_current_season_info", tool_names)
        self.assertIn("search_recent_results", tool_names)
        self.assertIn("google_search", tool_names)
        self.assertNotIn("search", tool_names)

    def test_instruction_enforces_last_event_without_year_policy(self):
        instruction = root_agent.static_instruction

        self.assertIn("google_search", instruction)
        self.assertIn("Relative expressions", instruction)

    def test_instruction_enforces_preseason_leader_guard(self):
        instruction = root_agent.static_instruction

        self.assertIn("Current year", instruction)
        self.assertIn("get_current_season_info", instruction)

    def test_before_model_callback_order(self):
        callback_names = [cb.__name__ for cb in root_agent.before_model_callback]

        self.assertLess(
            callback_names.index("inject_corrections"),
            callback_names.index("route_model"),
        )
        self.assertLess(
            callback_names.index("route_model"),
            callback_names.index("apply_grounding_policy"),
        )
        self.assertLess(
            callback_names.index("apply_grounding_policy"),
            callback_names.index("apply_response_contract"),
        )
        self.assertLess(
            callback_names.index("apply_response_contract"),
            callback_names.index("preflight_token_check"),
        )

    def test_after_model_callback_order(self):
        callback_names = [cb.__name__ for cb in root_agent.after_model_callback]

        self.assertIn("validate_structured_response", callback_names)
        self.assertIn("validate_grounding_outcome", callback_names)
        self.assertLess(
            callback_names.index("log_context_cache_metrics"),
            callback_names.index("validate_structured_response"),
        )
        self.assertLess(
            callback_names.index("validate_structured_response"),
            callback_names.index("validate_grounding_outcome"),
        )
        self.assertLess(
            callback_names.index("detect_corrections"),
            callback_names.index("store_cache"),
        )


if __name__ == "__main__":
    unittest.main()
