import unittest

from f1_agent.agent import root_agent


class AgentToolContractTests(unittest.TestCase):
    def test_instruction_uses_google_search_agent_name(self):
        instruction = root_agent.instruction

        self.assertIn("google_search_agent", instruction)
        self.assertNotIn("**google_search**", instruction)

    def test_instruction_includes_search_alias_recovery_rule(self):
        instruction = root_agent.instruction

        self.assertIn("invalid_tool_alias", instruction)
        self.assertIn("valid_tools", instruction)
        self.assertIn("Do NOT proactively choose `search`", instruction)

    def test_agent_registers_expected_tools(self):
        tool_names = []
        for tool in root_agent.tools:
            if hasattr(tool, "__name__"):
                tool_names.append(tool.__name__)
            elif hasattr(tool, "name"):
                tool_names.append(tool.name)

        self.assertIn("search_regulations", tool_names)
        self.assertIn("query_f1_history", tool_names)
        self.assertIn("search", tool_names)
        self.assertIn("google_search", tool_names)


if __name__ == "__main__":
    unittest.main()
