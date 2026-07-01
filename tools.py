from typing import List, Dict

from qdrant_store import QdrantVectorStore
from chunker import chunk_document

def add_to_knowledge_base(content: str, title: str, source: str) -> Dict:
    """
    Add a document to the knowledge base by chunking and upserting into Qdrant.
    """
    chunks = chunk_document(content, title, source)
    store = QdrantVectorStore()
    ids = store.upsert_chunks(chunks)
    return {"chunks_added": len(ids)}

def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a semantic search on the knowledge base.
    """
    store = QdrantVectorStore()
    results = store.search(query, k=max_results)
    return results
