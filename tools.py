from typing import List, Dict
from langchain.tools import tool

from init_vector_store import get_vector_store


@tool
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a semantic search in the knowledge base.

    Parameters:
    - query: The search query string.
    - max_results: Maximum number of results to return.

    Returns:
    A list of dictionaries with keys: content, title, source, score.
    """
    vector_store = get_vector_store()
    return vector_store.search(query, k=max_results)


@tool
def add_to_knowledge_base(content: str, title: str, source: str = "user") -> Dict:
    """
    Add a new document to the knowledge base.

    Parameters:
    - content: The full text content of the document.
    - title: The title of the document.
    - source: Optional source identifier (default: "user").

    Returns:
    A dictionary with the number of chunks added.
    """
    vector_store = get_vector_store()
    chunks_added = vector_store.add_document(content, title, source)
    return {"chunks_added": chunks_added}
