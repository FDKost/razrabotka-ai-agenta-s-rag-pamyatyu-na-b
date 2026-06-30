from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import QDRANT_URL, QDRANT_API_KEY, VECTOR_DIM, COLLECTION_NAME
from typing import List, Dict

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def create_collection():
    """
    Create the Qdrant collection if it does not exist.
    """
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=200,
                ef_search=200,
            ),
        )
    except Exception as e:
        # If collection already exists, ignore the error
        if "already exists" not in str(e):
            raise

def upsert_vectors(vectors: List[Dict]):
    """
    Upsert a list of vectors into the collection.
    Each vector dict should contain 'id', 'vector', and 'payload'.
    """
    points = []
    for vec in vectors:
        points.append(
            models.PointStruct(
                id=vec["id"],
                vector=vec["vector"],
                payload=vec["payload"],
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def query_vectors(vector: List[float], k: int = 5):
    """
    Query the collection for the top-k most similar vectors.
    Returns a list of search results.
    """
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=k,
        with_payload=True,
        with_vector=False,
    )
    return results
