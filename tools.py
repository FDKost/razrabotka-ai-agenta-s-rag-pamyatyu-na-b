from typing import List, Dict

from config import OLLAMA_MODEL
from chunker import chunk_document
from init_vector_store import get_vector_store

# Initialize the vector store once
store = get_vector_store()

def add_to_knowledge_base(content: str, title: str, source: str) -> Dict:
    """
    Add a document to the knowledge base by chunking and upserting into the vector store.
    Returns a dict with the number of chunks added.
    """
    chunks = chunk_document(content, title, source)
    ids = store.upsert_chunks(chunks)
    return {"chunks_added": len(ids)}

def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a semantic search on the knowledge base.
    Returns a list of results with content, title, source, and score.
    """
    return store.search(query, k=max_results)
