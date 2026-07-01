import os
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

from config import (
    QDRANT_URL,
    QDRANT_PORT,
    QDRANT_API_KEY,
    OLLAMA_MODEL,
)
from chunker import chunk_document

class QdrantVectorStoreWrapper:
    """
    Wrapper around langchain_qdrant.QdrantVectorStore to provide a simple API
    for adding documents and searching.
    """

    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=QDRANT_URL,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
        )
        self.embedding_function = OllamaEmbeddings(model=OLLAMA_MODEL)
        # Create or get collection
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_function,
        )

    def add_document(self, text: str, title: str, source: str) -> int:
        """
        Chunk the document, embed, and upsert into Qdrant.
        Returns the number of chunks added.
        """
        chunks = chunk_document(text, title, source)
        ids = []
        documents = []
        metadatas = []
        for chunk in chunks:
            ids.append(chunk["id"])
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "content": chunk["text"],
                }
            )
        self.vector_store.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        return len(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic search and return a list of matching chunks.
        Each result contains content, title, source, and score.
        """
        results = self.vector_store.similarity_search_with_score(query, k)
        hits = []
        for doc, score in results:
            metadata = doc.metadata or {}
            hits.append(
                {
                    "content": doc.page_content,
                    "title": metadata.get("title", ""),
                    "source": metadata.get("source", ""),
                    "score": score,
                }
            )
        return hits

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)


def get_vector_store() -> QdrantVectorStoreWrapper:
    """
    Factory function to return a singleton QdrantVectorStoreWrapper instance.
    """
    if not hasattr(get_vector_store, "_instance"):
        get_vector_store._instance = QdrantVectorStoreWrapper()
    return get_vector_store._instance
