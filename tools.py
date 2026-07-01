from typing import List, Dict
from langchain.tools import tool
from init_vector_store import get_vector_store

@tool
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a semantic search on the knowledge base.

    Parameters:
    - query: The search query string.
    - max_results: The maximum number of results to return.

    Returns a list of dictionaries with keys: content, title, source, score.
    """
    vector_store = get_vector_store()
    results = vector_store.search(query, k=max_results)
    return results

@tool
def add_to_knowledge_base(content: str, title: str, source: str) -> Dict:
    """
    Add a new document to the knowledge base.

    Parameters:
    - content: The full text content of the document.
    - title: The title of the document.
    - source: The source path or identifier of the document.

    Returns a dictionary with key 'chunks_added' indicating how many chunks were inserted.
    """
    vector_store = get_vector_store()
    chunks_added = vector_store.add_document(content, title, source)
    return {"chunks_added": chunks_added}
