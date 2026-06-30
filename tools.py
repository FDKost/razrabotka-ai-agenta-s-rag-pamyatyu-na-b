from typing import List, Dict

from qdrant_store import QdrantVectorStore
from chunker import chunk_text


def add_to_knowledge_base(content: str, title: str, source: str) -> Dict[str, int]:
    """
    Add the provided content to the knowledge base by chunking and storing each chunk.
    Returns a dictionary with the number of chunks added.
    """
    store = QdrantVectorStore()
    chunks = chunk_text(content, title, source)
    for chunk in chunks:
        store.add_document(chunk["content"], chunk["title"], chunk["source"])
    return {"chunks_added": len(chunks)}


def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the knowledge base for the given query and return matching chunks.
    """
    store = QdrantVectorStore()
    return store.search(query, max_results)
