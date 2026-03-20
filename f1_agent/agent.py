from google.adk.agents import Agent

from f1_agent.tools import search_regulations

root_agent = Agent(
    name="f1_regulations_assistant",
    model="gemini-2.0-flash",
    description="An AI assistant for the FIA 2026 Formula 1 Technical Regulations.",
    instruction="""You are an expert assistant on the FIA 2026 Formula 1 Technical Regulations.

    When a user asks about F1 regulations:
    1. Use the search_regulations tool to find relevant sections
    2. Provide a clear, accurate answer based on the retrieved content
    3. Always cite the specific articles and sections
    4. If the information is not found, say so honestly

    Be precise and technical when needed, but explain concepts clearly.
    Always respond based on the official FIA regulations document.""",
    tools=[search_regulations],
)
