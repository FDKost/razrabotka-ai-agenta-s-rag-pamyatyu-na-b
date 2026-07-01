from typing import List, Dict, Any
from langchain.tools import tool
from langchain_ollama import OllamaEmbeddings

from qdrant_store import QdrantVectorStore
from chunker import chunk_text

# Initialize the vector store
store = QdrantVectorStore()


@tool
def search_knowledge_base(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a semantic search on the knowledge base.

    Parameters:
    - query (str): The search query.
    - max_results (int): Maximum number of results to return.

    Returns:
    - List[Dict[str, Any]]: A list of matching chunks with keys:
      content, title, source, score.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string.")
    if not isinstance(max_results, int) or max_results <= 0:
        raise ValueError("max_results must be a positive integer.")

    results = store.search(query, max_results)
    return results


@tool
def add_to_knowledge_base(content: str, title: str, source: str) -> Dict[str, int]:
    """
    Add a new document or chunk to the knowledge base.

    Parameters:
    - content (str): The raw text content to add.
    - title (str): Title of the document.
    - source (str): Source path or identifier.

    Returns:
    - Dict[str, int]: Dictionary containing the number of chunks added.
    """
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string.")
    if not title or not isinstance(title, str):
        raise ValueError("Title must be a non-empty string.")
    if not source or not isinstance(source, str):
        raise ValueError("Source must be a non-empty string.")

    chunks = chunk_text(content, title, source)
    chunks_added = 0
    for chunk in chunks:
        store.add_document(
            content=chunk["content"],
            title=chunk["title"],
            source=chunk["source"],
        )
        chunks_added += 1

    return {"chunks_added": chunks_added}
