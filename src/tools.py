from langchain.tools import tool
from typing import List, Dict
from .qdrant_client import QdrantVectorStore
from .embeddings import get_embedding_model
import uuid

vector_store = QdrantVectorStore()
embedding_model = get_embedding_model()

@tool("search_knowledge_base")
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the knowledge base for relevant documents.

    Parameters
    ----------
    query : str
        The search query.
    max_results : int, optional
        Number of results to return (default 5).

    Returns
    -------
    str
        A formatted string containing the search results.
    """
    query_vector = embedding_model.embed_query(query)
    hits = vector_store.search(query_vector, k=max_results)
    if not hits:
        return "No relevant documents found."
    output_lines = []
    for hit in hits:
        meta = hit["metadata"]
        content = meta.get("content", "")
        title = meta.get("title", "Untitled")
        source = meta.get("source", "Unknown")
        output_lines.append(f"- {title} ({source}): {content[:200]}...")
    return "\n".join(output_lines)

@tool("add_to_knowledge_base")
def add_to_knowledge_base(content: str, title: str) -> str:
    """
    Add a new document to the knowledge base.

    Parameters
    ----------
    content : str
        The full text content of the document.
    title : str
        The title of the document.

    Returns
    -------
    str
        Confirmation message.
    """
    metadata = {"title": title, "source": "user_input"}
    chunks = []
    # Simple chunking: split by paragraphs
    for idx, paragraph in enumerate(content.split("\n\n")):
        chunk_text = paragraph.strip()
        if not chunk_text:
            continue
        embedding = embedding_model.embed_query(chunk_text)
        doc_id = str(uuid.uuid4())
        chunks.append(
            {
                "id": doc_id,
                "embedding": embedding,
                "metadata": {"content": chunk_text, "chunk_id": idx, "title": title},
            }
        )
    vector_store.add_documents(chunks)
    return f"Added {len(chunks)} chunks to the knowledge base."
