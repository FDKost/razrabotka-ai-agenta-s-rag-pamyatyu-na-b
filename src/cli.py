import click
from pathlib import Path
from .agent import agent_executor
from .ingestion import ingest_directory
from .tools import search_knowledge_base, add_to_knowledge_base
import sys

@click.group()
def cli():
    """Interactive RAG Agent CLI."""
    pass

@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def add(file_path):
    """Add a single file to the knowledge base."""
    path = Path(file_path)
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        click.echo(f"Error reading file: {e}")
        return
    result = add_to_knowledge_base(content=content, title=path.stem)
    click.echo(result)

@cli.command()
@click.argument("query", type=str)
@click.option("--max-results", default=5, help="Number of results to return")
def search(query, max_results):
    """Search the knowledge base."""
    result = search_knowledge_base(query=query, max_results=max_results)
    click.echo(result)

@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--overwrite", is_flag=True, help="Overwrite existing collection")
def ingest(directory, overwrite):
    """Ingest all files from a directory."""
    ingest_directory(Path(directory), overwrite=overwrite)
    click.echo("Ingestion complete.")

@cli.command()
def repl():
    """Start an interactive REPL."""
    click.echo("Starting RAG Agent REPL. Type /quit to exit.")
    while True:
        user_input = click.prompt("You")
        if user_input.strip() == "/quit":
            click.echo("Goodbye!")
            break
        elif user_input.startswith("/add"):
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                click.echo("Usage: /add <file_path>")
                continue
            file_path = parts[1]
            add.invoke(click.Context(add), file_path)
        elif user_input.startswith("/search"):
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                click.echo("Usage: /search <query>")
                continue
            query = parts[1]
            search.invoke(click.Context(search), query)
        else:
            result = agent_executor.invoke({"input": user_input})
            click.echo(result["output"])

if __name__ == "__main__":
    cli()
