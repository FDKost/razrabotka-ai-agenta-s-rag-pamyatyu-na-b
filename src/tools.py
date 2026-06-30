from langchain.tools import tool
from .qdrant_client import query_vectors, upsert_vectors
from .chunker import chunk_text
from .embeddings import get_embedding
from .config import COLLECTION_NAME

@tool("search_knowledge_base")
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the knowledge base for relevant chunks.
    """
    embedding = get_embedding(query)
    results = query_vectors(embedding, k=max_results)
    if not results:
        return "No relevant documents found."
    contents = []
    for res in results:
        payload = res.payload
        content = payload.get("content", "")
        source = payload.get("source", "")
        chunk_id = payload.get("chunk_id", "")
        contents.append(f"Source: {source} (chunk {chunk_id})\n{content}")
    return "\n\n".join(contents)

@tool("add_to_knowledge_base")
def add_to_knowledge_base(content: str, title: str) -> str:
    """
    Add a new document to the knowledge base.
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
    upsert_vectors(vectors)
    return f"Added {len(vectors)} chunks to the knowledge base."
