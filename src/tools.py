from .qdrant_client import QdrantVectorStore
from .embeddings import get_ollama_embeddings
from .chunker import chunk_text

def search_knowledge_base(query: str, max_results: int = 5):
    """
    Searches the knowledge base for the given query and returns formatted results.
    """
    store = QdrantVectorStore()
    embeddings_model = get_ollama_embeddings()
    query_emb = embeddings_model.embed_query(query)
    results = store.search(query_emb, limit=max_results)

    output = ""
    for r in results:
        payload = r.payload
        output += f"Source: {payload.get('source', 'unknown')}\n"
        output += f"Content: {payload.get('content', '')}\n\n"
    return output or "No results found."

def add_to_knowledge_base(content: str, title: str):
    """
    Adds a document (or content string) to the knowledge base.
    """
    store = QdrantVectorStore()
    embeddings_model = get_ollama_embeddings()
    chunks = chunk_text(content, {"source": title})
    contents = [c["content"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    embeddings = embeddings_model.embed_documents(contents)
    ids = [f"{title}_{c['metadata']['chunk_id']}" for c in chunks]
    store.add(embeddings, contents, ids=ids, metadata=metas)
    return f"Added {len(chunks)} chunks to the knowledge base."
