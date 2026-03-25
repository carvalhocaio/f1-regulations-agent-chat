run:
	uv run adk web

run-managed:
	uv run adk web --session_service_uri="agentengine://$$GOOGLE_CLOUD_AGENT_ENGINE_ID"
