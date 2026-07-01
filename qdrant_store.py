import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct, VectorParams, Distance

from langchain_ollama import OllamaEmbeddings

from config import QDRANT_URL, QDRANT_PORT, QDRANT_API_KEY, OLLAMA_MODEL

class QdrantVectorStore:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=QDRANT_URL,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
        )

        self.collection_name = "rag_collection"
        self.embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)

        # Determine embedding dimension
        sample_vector = self.embedding_model.embed_query("test")
        self.embedding_dim = len(sample_vector)

        # Create collection if it does not exist
        existing_collections = self.client.get_collections().collections
        if self.collection_name not in [c.name for c in existing_collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunks(self, chunks: List[Dict]) -> List[str]:
        """
        Upsert a list of chunk dictionaries into Qdrant.

        Each chunk dict must contain keys: id, text, title, source.
        Returns a list of point IDs that were inserted.
        """
        points = []
        for chunk in chunks:
            vector = self.embedding_model.embed_query(chunk["text"])
            payload = {
                "title": chunk["title"],
                "source": chunk["source"],
                "content": chunk["text"],
            }
            point = PointStruct(id=chunk["id"], vector=vector, payload=payload)
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return [p.id for p in points]

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic search and return a list of matching chunks.
        Each result contains content, title, source, and score.
        """
        query_vector = self.embedding_model.embed_query(query)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
            with_vector=False,
        )

        hits = []
        for hit in search_result:
            payload = hit.payload
            hits.append(
                {
                    "content": payload.get("content", ""),
                    "title": payload.get("title", ""),
                    "source": payload.get("source", ""),
                    "score": hit.score,
                }
            )
        return hits

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
