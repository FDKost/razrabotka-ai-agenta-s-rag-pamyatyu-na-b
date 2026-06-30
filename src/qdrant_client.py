from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME
from .embeddings import get_embedding_model
import uuid

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def _get_vector_size() -> int:
    try:
        return get_embedding_model().embedding_size
    except Exception:
        return 768

def create_collection(name: str = QDRANT_COLLECTION_NAME):
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=_get_vector_size(),
                distance=models.Distance.COSINE,
            ),
        )
    except Exception:
        # Collection may already exist; ignore
        pass

def upsert(name: str, ids: list, vectors: list, payloads: list):
    points = []
    for id_, vector, payload in zip(ids, vectors, payloads):
        points.append(
            models.PointStruct(
                id=id_,
                vector=vector,
                payload=payload,
            )
        )
    client.upsert(collection_name=name, points=points)

def search(name: str, query_vector: list, limit: int = 5):
    results = client.search(
        collection_name=name,
        query_vector=query_vector,
        limit=limit,
    )
    return results

def delete(name: str, ids: list):
    client.delete(
        collection_name=name,
        points_selector=models.PointSelector(
            points_selector=models.PointIdsSelector(
                points=ids
            )
        ),
    )
