import uuid
from typing import List, Dict, Any

import chromadb
from chromadb import PersistentClient
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION


class ChromaVectorStore:
    def __init__(self):
        # Initialize persistent client
        self.client = PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection_name = CHROMA_COLLECTION
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        # Create collection if it does not exist
        if self.collection_name not in self.client.list_collections():
            self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        self.collection = self.client.get_collection(self.collection_name)

    def add_document(
        self,
        content: str,
        title: str,
        source: str,
        chunk_id: str = None,
    ) -> str:
        """
        Add a single chunk to the vector store.

        Returns the point ID used in ChromaDB.
        """
        if chunk_id is None:
            chunk_id = str(uuid.uuid4())

        vector = self.embedding_model.embed_query(content)

        self.collection.add(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[content],
            metadatas=[{"title": title, "source": source, "content": content}],
        )
        return chunk_id

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic search and return a list of matching chunks.
        Each result contains content, title, source, and score.
        """
        query_vector = self.embedding_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=max_results,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "content": doc,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "score": 1 - dist,  # Convert cosine distance to similarity
                }
            )
        return hits

    def delete_collection(self):
        """
        Delete the entire collection. Use with caution.
        """
        self.client.delete_collection(self.collection_name)
