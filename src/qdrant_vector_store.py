import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME
from .embeddings import embed
from .chunker import chunk_text

logging.basicConfig(level=logging.INFO)

class QdrantVectorStore:
    """
    Wrapper around QdrantClient to handle collection creation,
    upsert, and search operations.
    """

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = QDRANT_COLLECTION_NAME
        if not self.client.collection_exists(self.collection_name):
            dummy_vector = embed("test")
            vector_size = len(dummy_vector)
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logging.info(f"Created collection {self.collection_name} with vector size {vector_size}")

    def add_documents_with_embeddings(self, content: str, title: str, source: str = None):
        """
        Chunk the content, embed each chunk, and upsert into Qdrant.
        """
        chunks = chunk_text(content, {"title": title, "source": source})
        points = []
        for chunk in chunks:
            vector = embed(chunk["content"])
            point_id = str(uuid.uuid4())
            payload = {
                "content": chunk["content"],
                "title": title,
                "source": source,
                "chunk_id": chunk["metadata"]["chunk_id"],
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logging.info(f"Upserted {len(points)} chunks for document '{title}'")

    def search(self, query: str, max_results: int = 5):
        """
        Search the collection for the most similar vectors.
        """
        query_vector = embed(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=max_results,
            with_payload=True,
            with_vectors=False,
        )
        formatted = []
        for res in results:
            payload = res.payload
            formatted.append({
                "content": payload.get("content"),
                "metadata": {
                    "title": payload.get("title"),
                    "source": payload.get("source"),
                    "chunk_id": payload.get("chunk_id"),
                },
                "score": res.score,
            })
        return formatted
