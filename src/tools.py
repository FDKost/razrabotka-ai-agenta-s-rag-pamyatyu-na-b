from langchain.tools import tool
from .qdrant_vector_store import QdrantVectorStore

# Singleton vector store instance
vector_store = QdrantVectorStore()

@tool("search_knowledge_base")
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the knowledge base for relevant chunks.
    """
    results = vector_store.search(query, max_results)
    if not results:
        return "No relevant documents found."
    formatted = "\n\n".join(
        [
            f"Title: {r['metadata'].get('title')}\n"
            f"Chunk ID: {r['metadata'].get('chunk_id')}\n"
            f"Source: {r['metadata'].get('source')}\n"
            f"Content: {r['content']}"
            for r in results
        ]
    )
    return formatted

@tool("add_to_knowledge_base")
def add_to_knowledge_base(content: str, title: str) -> str:
    """
    Add a document to the knowledge base.
    """
    vector_store.add_documents_with_embeddings(content, title)
    return f"Document '{title}' added to the knowledge base."
