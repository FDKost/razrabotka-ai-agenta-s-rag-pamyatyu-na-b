from langchain.tools import tool
from chroma_store import ChromaVectorStore
from chunker import chunk_text

# Initialize the vector store
store = ChromaVectorStore()


@tool
def search_knowledge_base(query: str, max_results: int = 5):
    """
    Perform a semantic search on the knowledge base.

    Returns a list of dictionaries with keys: content, title, source, score.
    """
    results = store.search(query, max_results)
    return results


@tool
def add_to_knowledge_base(content: str, title: str, source: str):
    """
    Add a new document or chunk to the knowledge base.

    The content is first split into chunks using the configured chunker.
    Returns a dictionary with the number of chunks added.
    """
    chunks = chunk_text(content, title, source)
    for chunk in chunks:
        store.add_document(chunk["content"], chunk["title"], chunk["source"])
    return {"chunks_added": len(chunks)}
