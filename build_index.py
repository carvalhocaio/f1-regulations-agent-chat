"""
Script to build the FAISS vector store from the PDFs.
Run this once before starting the chat:

    uv run build_index.py
"""

from rich.console import Console

console = Console()


def main() -> None:
    from f1_agent.rag import build_vector_store

    console.print("[bold]🔧 Building vector store from PDFs...[/bold]")
    console.print("[dim]This may take a minute on first run.[/dim]\n")

    try:
        build_vector_store()
        console.print(
            "[bold green]✅ Vector store built successfully![/bold green]\n"
        )
    except FileNotFoundError as e:
        console.print(f"[bold red]❌ {e}[/bold red]\n")

    console.print("[dim]You can now run: python main.py[/dim]")


if __name__ == "__main__":
    main()
