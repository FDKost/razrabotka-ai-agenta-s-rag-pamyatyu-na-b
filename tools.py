import json
from typing import List, Dict

from langchain.tools import tool
from qdrant_store import QdrantVectorStore
from chunker import chunk_document

@tool
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a semantic search on the knowledge base.

    Parameters:
    - query: The search query string.
    - max_results: Number of top results to return.

    Returns a list of dictionaries with keys: content, title, source, score.
    """
    store = QdrantVectorStore()
    results = store.search(query, k=max_results)
    return results

@tool
def add_to_knowledge_base(content: str, title: str, source: str) -> Dict:
    """
    Add a new document to the knowledge base.

    Parameters:
    - content: Raw text content of the document.
    - title: Title of the document.
    - source: Source path or identifier.

    Returns a dictionary with the number of chunks added.
    """
    chunks = chunk_document(content, title, source)
    store = QdrantVectorStore()
    inserted_ids = store.upsert_chunks(chunks)
    return {"chunks_added": len(inserted_ids), "chunk_ids": inserted_ids}
