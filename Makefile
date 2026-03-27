run:
	@printf "ADK local web UI was removed with Spanner-linked dependencies.\n"
	@printf "Use project tests for validation: uv run python -m unittest discover tests -v\n"

run-managed:
	@printf "Managed sessions were removed. This project now uses in-memory sessions only.\n"
