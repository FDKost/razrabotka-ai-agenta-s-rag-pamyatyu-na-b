from qdrant_client import QdrantClient as Qdrant
from qdrant_client.http import models
from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

class QdrantVectorStore:
    """
    Wrapper around QdrantClient to handle collection creation,
    document insertion, and similarity search.
    """

    def __init__(self):
        self.client = Qdrant(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = QDRANT_COLLECTION_NAME
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections()
        existing = [c.name for c in collections.collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # default size for Ollama embeddings
                    distance=models.Distance.COSINE
                ),
            )

    def add(self, embeddings, documents, ids=None, metadata=None):
        """
        Adds documents to the collection.
        """
        points = []
        for i, (emb, doc, meta) in enumerate(zip(embeddings, documents, metadata)):
            point_id = ids[i] if ids else i
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=meta
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding, limit=5):
        """
        Performs a similarity search.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
        )
        return results
