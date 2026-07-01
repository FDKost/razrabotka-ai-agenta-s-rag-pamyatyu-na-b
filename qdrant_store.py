import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct, VectorParams, Distance

from langchain_ollama import OllamaEmbeddings

from config import QDRANT_URL, QDRANT_PORT, QDRANT_API_KEY


class QdrantVectorStore:
    def __init__(self):
        # Initialize Qdrant client with retry logic
        try:
            self.client = QdrantClient(
                url=QDRANT_URL,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant at {QDRANT_URL}:{QDRANT_PORT} - {e}")

        self.collection_name = "rag_collection"
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        # Determine embedding dimension
        sample_vector = self.embedding_model.embed_query("test")
        self.embedding_dim = len(sample_vector)

        # Create collection if it does not exist
        existing_collections = self.client.get_collections().collections
        if self.collection_name not in [c.name for c in existing_collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
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

        payload = {"title": title, "source": source, "content": content}

        point = PointStruct(id=chunk_id, vector=vector, payload=payload)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        return chunk_id

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
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

        hits = []
        for hit in search_result:
            payload = hit.payload
            hits.append(
                {
                    "content": payload.get("content", ""),
                    "title": payload.get("title", ""),
                    "source": payload.get("source", ""),
                    "score": hit.score,  # Qdrant returns similarity score
                }
            )
        return hits

    def delete_collection(self):
        """
        Delete the entire collection. Use with caution.
        """
        self.client.delete_collection(self.collection_name)
