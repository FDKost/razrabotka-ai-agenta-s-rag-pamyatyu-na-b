from langchain.tools import tool
from src.qdrant_client import search, upsert_vectors, create_collection
from src.embeddings import get_embedding
from src.chunker import chunk_text
from pathlib import Path

@tool
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the knowledge base for the most relevant chunks to the query.
    Returns a JSON string with the results.
    """
    query_vector = get_embedding(query)
    results = search(query_vector, limit=max_results)
    output = []
    for r in results:
        payload = r.payload
        output.append(
            {
                "content": payload.get("content"),
                "source": payload.get("source"),
                "chunk_id": payload.get("chunk_id"),
                "score": r.score,
            }
        )
    import json
    return json.dumps(output, indent=2)

@tool
def add_to_knowledge_base(content: str, title: str) -> str:
    """
    Add a new document to the knowledge base.
    The content is split into chunks and embedded.
    Returns a confirmation string.
    """
    create_collection()
    metadata = {"source": title}
    chunks = chunk_text(content, metadata)
    vectors = []
    for chunk in chunks:
        vector = get_embedding(chunk["content"])
        point_id = f"{title}_{chunk['metadata']['chunk_id']}"
        vectors.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": {
                    "content": chunk["content"],
                    "source": title,
                    "chunk_id": chunk["metadata"]["chunk_id"],
                },
            }
        )
    upsert_vectors(vectors)
    return f"Added {len(vectors)} chunks from '{title}'."
