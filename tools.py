from langchain.tools import tool
from init_vector_store import get_vector_store
from chunker import chunk_document


@tool
def search_knowledge_base(query: str, max_results: int = 5):
    """
    Perform a semantic search on the knowledge base.
    """
    store = get_vector_store()
    return store.search(query, k=max_results)


@tool
def add_to_knowledge_base(content: str, title: str, source: str = None):
    """
    Add a new document or chunk to the knowledge base.
    """
    store = get_vector_store()
    chunks = chunk_document(content, title, source or "")
    ids = store.upsert_chunks(chunks)
    return {"chunks_added": len(ids), "ids": ids}
