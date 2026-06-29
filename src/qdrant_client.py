from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    VECTOR_DIM,
    COLLECTION_NAME,
)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_collection():
    existing = client.get_collections()
    if COLLECTION_NAME not in [c.name for c in existing.collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM, distance=models.Distance.COSINE
            ),
        )


def upsert_vectors(vectors):
    """
    vectors: list of dicts with keys:
        id: str
        vector: list[float]
        payload: dict
    """
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=v["id"],
                vector=v["vector"],
                payload=v["payload"],
            )
            for v in vectors
        ],
    )


def search(query_vector, limit=5):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
    )
    return results
