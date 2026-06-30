from qdrant_client import QdrantClient, models
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")

class QdrantVectorStore:
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
        )
        self.collection_name = QDRANT_COLLECTION_NAME
        # Ensure collection exists
        if self.collection_name not in self.client.get_collections().collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # default size for Ollama embeddings; adjust if needed
                    distance=models.Distance.COSINE,
                ),
            )

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        documents: list of dicts with keys:
            id: str
            embedding: List[float]
            metadata: dict
        """
        vectors = [
            models.PointStruct(
                id=doc["id"],
                vector=doc["embedding"],
                payload=doc["metadata"],
            )
            for doc in documents
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=vectors,
        )

    def search(self, query_vector: List[float], k: int = 5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
            with_vector=False,
        )
        hits = []
        for hit in results:
            hits.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "metadata": hit.payload,
                    "embedding": hit.vector,
                }
            )
        return hits

    def delete(self, ids: List[str]):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointSelector(
                points=ids
            ),
        )

    def get_collection(self):
        return self.client.get_collection(self.collection_name)

    def close(self):
        self.client.close()
