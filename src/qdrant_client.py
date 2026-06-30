import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import QdrantException
from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

class QdrantClientWrapper:
    """
    Wrapper around QdrantClient to simplify collection management and vector operations.
    """

    def __init__(self, url: str = QDRANT_URL, api_key: str = QDRANT_API_KEY, collection_name: str = QDRANT_COLLECTION_NAME):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Ensure the collection exists; create it if it does not.
        """
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Ollama embeddings size
                    distance=models.Distance.COSINE,
                ),
            )
        except QdrantException as e:
            # If collection already exists, ignore
            if "already exists" not in str(e):
                raise

    def upsert(self, vectors: List[Dict[str, Any]]):
        """
        Upsert a list of vectors into the collection.

        Each vector dict should contain:
            - id: str
            - vector: List[float]
            - payload: dict
        """
        points = [
            models.PointStruct(
                id=vec["id"],
                vector=vec["vector"],
                payload=vec["payload"],
            )
            for vec in vectors
        ]
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except QdrantException as e:
            raise RuntimeError(f"Failed to upsert vectors: {e}")

    def similarity_search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a similarity search and return the top results.

        Returns a list of dicts containing:
            - id
            - score
            - payload
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=0.0,
            )
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in results
            ]
        except QdrantException as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}")

    def delete_collection(self):
        """
        Delete the entire collection.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except QdrantException as e:
            raise RuntimeError(f"Failed to delete collection: {e}")
