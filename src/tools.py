from langchain.tools import tool
from .qdrant_client import qdrant
from .embeddings import embed
from .chunker import chunk_text
import json

@tool(
    name="search_knowledge_base",
    description="Search the vector store for relevant documents based on a query."
)
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the knowledge base for documents relevant to the query.

    Returns a JSON string containing the results and metadata.
    """
    query_vector = embed(query)
    results = qdrant.search(query_vector, limit=max_results)
    # Convert results to a serializable form
    formatted = []
    for res in results:
        formatted.append(
            {
                "id": res.id,
                "score": res.score,
                "payload": res.payload,
            }
        )
    return json.dumps({"results": formatted}, indent=2)

@tool(
    name="add_to_knowledge_base",
    description="Add a new document to the vector store by ingesting its content."
)
def add_to_knowledge_base(content: str, title: str) -> str:
    """
    Ingest a document into the knowledge base.

    The document is split into chunks, each chunk is embedded, and the
    resulting vectors are upserted into Qdrant.
    """
    # Prepare metadata
    metadata = {"title": title}
    chunks = chunk_text(content, metadata)
    vectors = [embed(chunk["content"]) for chunk in chunks]
    payloads = [chunk["metadata"] for chunk in chunks]
    qdrant.upsert(vectors, payloads)
    return f"Document '{title}' ingested with {len(chunks)} chunks."
