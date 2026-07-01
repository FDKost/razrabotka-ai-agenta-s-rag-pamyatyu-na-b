from typing import List, Dict

from chroma_store import ChromaVectorStore
from chunker import chunk_text


def add_to_knowledge_base(content: str, title: str, source: str) -> Dict[str, int]:
    """
    Add the provided content to the knowledge base.

    Returns a dictionary with the number of chunks added.
    """
    store = ChromaVectorStore()
    chunks = chunk_text(content, title, source)
    for chunk in chunks:
        store.add_document(
            content=chunk["content"],
            title=chunk["title"],
            source=chunk["source"],
        )
    return {"chunks_added": len(chunks)}


def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the knowledge base for the given query.

    Returns a list of matching chunks.
    """
    store = ChromaVectorStore()
    return store.search(query, max_results=max_results)
