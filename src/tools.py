from langchain.tools import tool
from .ingestion import add_document_from_content
from .qdrant_client import search
from .config import QDRANT_COLLECTION_NAME
from .embeddings import get_embedding_model

@tool
def add_to_knowledge_base(content: str, title: str) -> str:
    """
    Adds a document to the knowledge base.
    """
    add_document_from_content(content, title)
    return f"Document '{title}' added to the knowledge base."

@tool
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Searches the knowledge base for the query and returns top results.
    """
    embeddings = get_embedding_model()
    query_vector = embeddings.embed_query(query)
    results = search(QDRANT_COLLECTION_NAME, query_vector, limit=max_results)
    if not results:
        return "No relevant documents found."
    output_lines = []
    for idx, res in enumerate(results, start=1):
        payload = res.payload
        title = payload.get("title", "Untitled")
        chunk_id = payload.get("chunk_id", "N/A")
        score = res.score
        output_lines.append(f"{idx}. [{title} - chunk {chunk_id}] (score: {score:.4f})")
    return "\n".join(output_lines)
