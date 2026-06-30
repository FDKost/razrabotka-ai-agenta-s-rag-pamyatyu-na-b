import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from langchain_ollama import OllamaEmbeddings

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION


class QdrantVectorStore:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = QDRANT_COLLECTION
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        # Ensure collection exists
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedding_model.get_num_dimensions(),
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

    def add_document(
        self,
        content: str,
        title: str,
        source: str,
        chunk_id: str = None,
    ) -> str:
        """
        Add a single chunk to the vector store.

        Returns the point ID used in Qdrant.
        """
        if chunk_id is None:
            chunk_id = str(uuid.uuid4())

        vector = self.embedding_model.embed_query(content)

        payload = {
            "title": title,
            "source": source,
            "content": content,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qdrant_models.PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return chunk_id

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic search and return a list of matching chunks.
        Each result contains content, title, source, and score.
        """
        query_vector = self.embedding_model.embed_query(query)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=max_results,
            with_payload=True,
            with_vector=False,
        )

        results = []
        for hit in search_result:
            payload = hit.payload
            results.append(
                {
                    "content": payload.get("content", ""),
                    "title": payload.get("title", ""),
                    "source": payload.get("source", ""),
                    "score": hit.score,
                }
            )
        return results

    def delete_collection(self):
        """
        Delete the entire collection. Use with caution.
        """
        self.client.delete_collection(self.collection_name)
