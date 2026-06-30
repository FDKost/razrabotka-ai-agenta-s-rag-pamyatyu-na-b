import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import get_config
from .embeddings import get_embedding_model

class QdrantClientWrapper:
    def __init__(self):
        cfg = get_config()
        self.client = QdrantClient(url=cfg["qdrant_url"], api_key=cfg["qdrant_api_key"])
        self.collection_name = cfg["qdrant_collection_name"]
        self.embedding_model = get_embedding_model()

        # Determine vector size using a dummy embedding
        dummy_embedding = self.embedding_model.embed_query("test")
        vector_size = len(dummy_embedding)

        # Ensure collection exists
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def add_documents(self, documents: List[Dict]):
        """
        documents: list of dicts with keys:
            - content: str
            - metadata: dict
        """
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.embed_documents(contents)

        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id = str(uuid.uuid4())
            payload = doc.get("metadata", {})
            payload["content"] = doc["content"]
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self, query: str, limit: int = 5):
        embedding = self.embedding_model.embed_query(query)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit,
        )
        results = []
        for hit in search_result:
            payload = hit.payload
            content = payload.get("content", "")
            results.append({"content": content, "metadata": payload})
        return results
