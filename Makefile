.PHONY: dev api

dev:
	uv run adk web

api:
	uv run adk api_server --host 127.0.0.1 --port 8080 --session_service_uri memory:// --auto_create_session .
