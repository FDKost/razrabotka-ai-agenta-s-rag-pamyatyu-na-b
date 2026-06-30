from typing import List, Dict, Any

from langchain.tools import tool

from qdrant_store import QdrantVectorStore
from chunker import chunk_text

# Instantiate the vector store once
vector_store = QdrantVectorStore()


@tool("search_knowledge_base")
def search_knowledge_base(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Perform a semantic search on the knowledge base.

    Parameters
    ----------
    query : str
        The natural language query to search for.
    max_results : int, optional
        Number of top results to return (default 5).

    Returns
    -------
    List[Dict[str, Any]]
        A list of matching chunks with content, title, source, and score.
    """
    if not query:
        raise ValueError("Query must not be empty.")
    if max_results <= 0:
        raise ValueError("max_results must be a positive integer.")
    return vector_store.search(query, max_results)


@tool("add_to_knowledge_base")
def add_to_knowledge_base(
    content: str,
    title: str,
    source: str,
) -> Dict[str, Any]:
    """
    Add a new document (or chunk) to the knowledge base.

    Parameters
    ----------
    content : str
        The text content to add.
    title : str
        Title or identifier for the document.
    source : str
        Source path or description.

    Returns
    -------
    Dict[str, Any]
        Summary of the operation, including number of chunks added.
    """
    if not content:
        raise ValueError("Content must not be empty.")
    if not title:
        raise ValueError("Title must not be empty.")
    if not source:
        raise ValueError("Source must not be empty.")

    # Chunk the content
    chunks = chunk_text(content, title, source)
    added_ids = []
    for chunk in chunks:
        chunk_id = vector_store.add_document(
            content=chunk["content"],
            title=chunk["title"],
            source=chunk["source"],
        )
        added_ids.append(chunk_id)

    return {
        "title": title,
        "source": source,
        "chunks_added": len(chunks),
        "point_ids": added_ids,
    }
