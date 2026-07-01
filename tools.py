import json
from typing import List, Dict

from langchain.tools import tool

from init_vector_store import get_vector_store

@tool
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a semantic search in the knowledge base.

    Parameters
    ----------
    query : str
        The query string to search for.
    max_results : int, optional
        Number of results to return (default 5).

    Returns
    -------
    List[Dict]
        List of search results with keys: content, title, source, score.
    """
    store = get_vector_store()
    return store.search(query, k=max_results)

@tool
def add_to_knowledge_base(content: str, title: str, source: str) -> Dict:
    """
    Add a document to the knowledge base.

    Parameters
    ----------
    content : str
        The full text content of the document.
    title : str
        The title of the document.
    source : str
        The source path or identifier.

    Returns
    -------
    Dict
        Dictionary containing the number of chunks added.
    """
    store = get_vector_store()
    chunks_added = store.add_document(content, title, source)
    return {"chunks_added": chunks_added}
