import click
from pathlib import Path
from src.agent import agent_executor
from src.ingestion import ingest_directory
import json

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
    result = agent_executor.invoke(
        {
            "input": f"add_to_knowledge_base(content={content!r}, title={path.stem!r})",
        }
    )
    click.echo(result["output"])

@cli.command()
@click.argument("query", type=str)
def search(query):
    """Search the knowledge base."""
    result = agent_executor.invoke(
        {
            "input": f"search_knowledge_base(query={query!r}, max_results=5)",
        }
    )
    click.echo(result["output"])

@cli.command()
@click.argument("directory", type=click.Path(exists=True))
def ingest(directory):
    """Ingest all files from a directory."""
    ingest_directory(Path(directory))
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
            _, file_path = user_input.split(maxsplit=1)
            add.invoke(click.Context(add), file_path)
        elif user_input.startswith("/search"):
            _, query = user_input.split(maxsplit=1)
            search.invoke(click.Context(search), query)
        else:
            result = agent_executor.invoke({"input": user_input})
            click.echo(result["output"])

if __name__ == "__main__":
    cli()
