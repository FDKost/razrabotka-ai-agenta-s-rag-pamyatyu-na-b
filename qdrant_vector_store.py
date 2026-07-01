import os
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    OLLAMA_EMBED_MODEL,
)
from chunker import chunk_document


class QdrantVectorStoreWrapper:
    """
    Wrapper around langchain's Qdrant vector store to provide a simple API
    for adding documents and searching.
    """

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.embedding_function = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

        # Create collection if it does not exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # default size for Ollama embeddings
                    distance=Distance.COSINE,
                ),
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def add_document(self, text: str, title: str, source: str) -> int:
        """
        Chunk the document, embed, and upsert into Qdrant.
        Returns the number of chunks added.
        """
        chunks = chunk_document(text, title, source)
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "content": chunk["text"],
                },
            )
            documents.append(doc)
        self.vector_store.add_documents(documents)
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

    def persist(self):
        """Persist the collection to disk (Qdrant auto-persist)."""
        self.client.flush()
