"""
Script to build the FAISS vector store and SQLite database.
Run this once before starting the chat:

    uv run build_index.py
"""

from rich.console import Console

console = Console()


def main() -> None:
    from f1_agent.db import build_database
    from f1_agent.rag import build_vector_store

    console.print("[bold]Building vector store from PDFs...[/bold]")
    console.print("[dim]This may take a minute on first run.[/dim]\n")

    try:
        build_vector_store()
        console.print("[bold green]Vector store built successfully![/bold green]\n")
    except FileNotFoundError as e:
        console.print(f"[bold red]{e}[/bold red]\n")

    console.print("[bold]Building SQLite database from CSVs...[/bold]")

    try:
        build_database()
        console.print("[bold green]SQLite database built![/bold green]\n")
    except FileNotFoundError as e:
        console.print(f"[bold red]{e}[/bold red]\n")

    console.print(
        "[dim]Artifacts ready. Run tests with: uv run python -m unittest discover tests -v[/dim]"
    )


if __name__ == "__main__":
    main()
