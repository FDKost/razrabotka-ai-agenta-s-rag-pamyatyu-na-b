import click

from tools import add_to_knowledge_base, search_knowledge_base


@click.group()
def cli():
    """Simple CLI for interacting with the RAG agent."""
    pass


@cli.command()
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
def add(file_path):
    """Add a text file to the knowledge base."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    title = Path(file_path).stem
    source = Path(file_path).resolve()
    result = add_to_knowledge_base(content, title, str(source))
    click.echo(f"Added '{title}' from '{source}'. Chunks added: {result['chunks_added']}")


@cli.command()
@click.argument("query", nargs=-1)
def search(query):
    """Search the knowledge base."""
    query_str = " ".join(query)
    results = search_knowledge_base(query_str, max_results=5)
    if not results:
        click.echo("No results found.")
        return
    for idx, res in enumerate(results, 1):
        click.echo(f"\nResult {idx}:")
        click.echo(f"Title: {res['title']}")
        click.echo(f"Source: {res['source']}")
        click.echo(f"Score: {res['score']:.4f}")
        click.echo(f"Content: {res['content'][:500]}{'...' if len(res['content']) > 500 else ''}")


@cli.command()
def quit():
    """Exit the CLI."""
    click.echo("Goodbye!")
    raise SystemExit


if __name__ == "__main__":
    cli()
