from langchain.tools import tool
from .chromadb_client import similarity_search
from .embeddings import get_embedding
from .chunker import chunk_text
from .ingestion import ingest_directory
import os
from pathlib import Path

@tool
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the knowledge base for relevant chunks.
    Returns a formatted string of results.
    """
    query_vector = get_embedding(query)
    hits = similarity_search(query_vector, max_results)
    if not hits:
        return "No relevant results found."
    output_lines = []
    for hit in hits:
        meta = hit["metadata"]
        output_lines.append(
            f"Score: {hit['score']:.4f} | Source: {meta.get('source', 'unknown')} | Chunk ID: {meta.get('chunk_id')}\nContent: {meta.get('content')[:200]}..."
        )
    return "\n\n".join(output_lines)

@tool
def add_to_knowledge_base(content: str, title: str) -> str:
    """
    Add a single document to the knowledge base.
    """
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
                    "source": metadata["source"],
                    "chunk_id": chunk["metadata"]["chunk_id"],
                },
            }
        )
    from .chromadb_client import upsert_vectors
    upsert_vectors(vectors)
    return f"Added {len(vectors)} chunks to the knowledge base."
