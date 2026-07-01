from typing import List, Dict

from langchain.tools import tool

from qdrant_store import QdrantVectorStore
from chunker import chunk_document


@tool
def search_knowledge_base(query: str, max_results: int) -> List[Dict]:
    """
    Perform a semantic search on the knowledge base.

    Parameters
    ----------
    query : str
        The search query.
    max_results : int
        The maximum number of results to return.

    Returns
    -------
    List[Dict]
        A list of matching documents, each containing content, title, source, and score.
    """
    store = QdrantVectorStore()
    return store.search(query, k=max_results)


@tool
def add_to_knowledge_base(content: str, title: str, source: str = None) -> Dict:
    """
    Add a new document or chunk to the knowledge base.

    Parameters
    ----------
    content : str
        The full text content of the document.
    title : str
        The title of the document.
    source : str, optional
        The source path or identifier of the document.

    Returns
    -------
    Dict
        A dictionary containing the number of chunks added.
    """
    store = QdrantVectorStore()
    # If source is not provided, use a placeholder
    if source is None:
        source = "unknown"

    chunks = chunk_document(content, title, source)
    inserted_ids = store.upsert_chunks(chunks)
    return {"chunks_added": len(inserted_ids), "inserted_ids": inserted_ids, "title": title, "source": source}
