from typing import List, Dict

from langchain.tools import tool
from qdrant_store import QdrantVectorStore
from chunker import chunk_text


@tool(name="search_knowledge_base", description="Search the knowledge base for relevant documents.")
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict[str, any]]:
    """
    Perform a semantic search on the knowledge base.

    Parameters
    ----------
    query : str
        The search query.
    max_results : int, optional
        Maximum number of results to return. Defaults to 5.

    Returns
    -------
    List[Dict[str, any]]
        A list of search results, each containing content, title, source, and score.
    """
    store = QdrantVectorStore()
    results = store.search(query, max_results=max_results)
    return results


@tool(name="add_to_knowledge_base", description="Add a new document or chunk to the knowledge base.")
def add_to_knowledge_base(content: str, title: str, source: str) -> Dict[str, any]:
    """
    Add a document to the knowledge base by chunking it and storing each chunk.

    Parameters
    ----------
    content : str
        The full text content of the document.
    title : str
        Title or identifier for the document.
    source : str
        Source path or URL of the document.

    Returns
    -------
    Dict[str, any]
        Dictionary containing the number of chunks added.
    """
    store = QdrantVectorStore()
    chunks = chunk_text(content, title, source)
    chunks_added = 0
    for chunk in chunks:
        store.add_document(
            content=chunk["content"],
            title=chunk["title"],
            source=chunk["source"],
        )
        chunks_added += 1
    return {"chunks_added": chunks_added}
