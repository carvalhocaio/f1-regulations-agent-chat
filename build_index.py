"""
Script to build the FAISS vector store and SQLite database.
Run this once before starting the chat:

    uv run build_index.py
"""

from rich.console import Console

console = Console()


def main() -> int:
    from f1_agent.db import build_database
    from f1_agent.rag import build_vector_store

    all_ok = True

    console.print("[bold]Building vector store from PDFs...[/bold]")
    console.print("[dim]This may take a minute on first run.[/dim]\n")

    try:
        build_vector_store()
        vector_store_ok = True
        console.print("[bold green]Vector store built successfully![/bold green]\n")
    except (FileNotFoundError, ValueError) as e:
        vector_store_ok = False
        all_ok = False
        console.print(f"[bold red]{e}[/bold red]\n")
    except Exception as e:
        vector_store_ok = False
        all_ok = False
        console.print(
            f"[bold red]Unexpected error building vector store: {e}[/bold red]"
        )
        console.print_exception()
        console.print()

    console.print("[bold]Building SQLite database from CSVs...[/bold]")

    try:
        build_database()
        sqlite_ok = True
        console.print("[bold green]SQLite database built![/bold green]\n")
    except FileNotFoundError as e:
        sqlite_ok = False
        all_ok = False
        console.print(f"[bold red]{e}[/bold red]\n")
    except Exception as e:
        sqlite_ok = False
        all_ok = False
        console.print(
            f"[bold red]Unexpected error building SQLite database: {e}[/bold red]"
        )
        console.print_exception()
        console.print()

    console.print("[bold]Build summary[/bold]")
    console.print(
        f"- Vector store: {'[green]OK[/green]' if vector_store_ok else '[red]FAIL[/red]'}"
    )
    console.print(
        f"- SQLite DB: {'[green]OK[/green]' if sqlite_ok else '[red]FAIL[/red]'}"
    )

    if all_ok:
        console.print(
            "\n[dim]Artifacts ready. Run tests with: uv run python -m unittest discover tests -v[/dim]"
        )
        return 0

    console.print("\n[bold red]Build completed with failures.[/bold red]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
