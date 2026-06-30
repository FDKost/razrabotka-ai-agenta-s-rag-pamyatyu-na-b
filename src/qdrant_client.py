from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME
from .embeddings import embed

class QdrantWrapper:
    """
    Simple wrapper around QdrantClient to handle collection creation,
    upsert, and search operations.
    """
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection = QDRANT_COLLECTION_NAME
        # Ensure collection exists
        if not self.client.collection_exists(self.collection):
            # Determine vector size from a dummy embedding
            dummy_vector = embed("test")
            vector_size = len(dummy_vector)
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(self, vectors, payloads):
        """
        Upsert a batch of vectors with associated payloads.
        """
        batch = models.Batch(
            ids=[i for i in range(len(vectors))],
            vectors=vectors,
            payload=payloads,
        )
        self.client.upsert(
            collection_name=self.collection,
            points=batch,
        )

    def search(self, query_vector, limit=5):
        """
        Search the collection for the most similar vectors.
        """
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return results

# Singleton instance
qdrant = QdrantWrapper()
