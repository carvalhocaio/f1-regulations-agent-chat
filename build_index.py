"""
Script to build the FAISS vector store from the PDF.
Run this once before starting the chat:

    uv run build_index.py
"""

from rich.console import Console

console = Console()


def main() -> None:
    from f1_agent.rag import PDF_FILES, build_vector_store

    console.print("[bold]🔧 Building vector stores from PDFs...[/bold]")
    console.print("[dim]This may take a minute on first run.[/dim]\n")

    for year in PDF_FILES:
        try:
            console.print(f"[bold]📄 Building index for {year} regulations...[/bold]")
            build_vector_store(year=year)
            console.print(
                f"[bold green]✅ {year} vector store built successfully![/bold green]\n"
            )
        except FileNotFoundError as e:
            console.print(f"[bold red]❌ {e}[/bold red]\n")

    console.print("[dim]You can now run: python main.py[/dim]")


if __name__ == "__main__":
    main()
