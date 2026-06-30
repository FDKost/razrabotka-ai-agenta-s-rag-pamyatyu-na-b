import uuid
from typing import List, Dict, Any
from langchain.tools import tool
from .qdrant_client import QdrantClientWrapper
from .embeddings import get_embedding_model
from .config import QDRANT_COLLECTION_NAME

@tool
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the knowledge base for the most relevant chunks.

    Parameters
    ----------
    query : str
        The search query.
    max_results : int, optional
        Number of top results to return (default 5).

    Returns
    -------
    str
        A formatted string containing the top results.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string.")
    if not isinstance(max_results, int) or max_results <= 0:
        raise ValueError("max_results must be a positive integer.")

    client = QdrantClientWrapper()
    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(query)

    results = client.similarity_search(query_vector, limit=max_results)

    if not results:
        return "No relevant results found."

    formatted = []
    for res in results:
        payload = res["payload"]
        title = payload.get("title", "Unknown")
        chunk_id = payload.get("chunk_id", "N/A")
        score = res["score"]
        snippet = payload.get("content", "")[:200] + ("..." if len(payload.get("content", "")) > 200 else "")
        formatted.append(f"Title: {title}\nChunk ID: {chunk_id}\nScore: {score:.4f}\nSnippet: {snippet}\n---")

    return "\n".join(formatted)

@tool
def add_to_knowledge_base(content: str, title: str) -> str:
    """
    Add a new document to the knowledge base.

    Parameters
    ----------
    content : str
        The full text content of the document.
    title : str
        A title for the document.

    Returns
    -------
    str
        Confirmation message.
    """
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string.")
    if not title or not isinstance(title, str):
        raise ValueError("Title must be a non-empty string.")

    client = QdrantClientWrapper()
    embedding_model = get_embedding_model()

    # Chunk the content
    from .chunker import chunk_text
    chunks = chunk_text(content, {"title": title, "source": "manual_addition"})
    embeddings = embedding_model.embed_documents([c["content"] for c in chunks])

    vectors = []
    for idx, (chunk, embed) in enumerate(zip(chunks, embeddings)):
        vector = {
            "id": str(uuid.uuid4()),
            "vector": embed,
            "payload": {**chunk["metadata"]},
        }
        vectors.append(vector)

    client.upsert(vectors)
    return f"Document '{title}' added with {len(chunks)} chunks."
