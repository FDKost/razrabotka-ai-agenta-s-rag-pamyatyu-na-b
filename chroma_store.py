import uuid
from typing import List, Dict, Any

from chromadb import Client
from langchain_ollama import OllamaEmbeddings

from config import OLLAMA_MODEL

class ChromaVectorStore:
    def __init__(self):
        # Initialize Chroma client
        self.client = Client()
        self.collection_name = "rag_collection"
        self.embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)

        # Create or get collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_model,
        )

    def upsert_chunks(self, chunks: List[Dict]) -> List[str]:
        """
        Upsert a list of chunk dictionaries into Chroma.

        Each chunk dict must contain keys: id, text, title, source.
        Returns a list of point IDs that were inserted.
        """
        ids = []
        documents = []
        metadatas = []
        for chunk in chunks:
            ids.append(chunk["id"])
            documents.append(chunk["text"])
            metadatas.append({
                "title": chunk["title"],
                "source": chunk["source"],
                "content": chunk["text"],
            })
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        return ids

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic search and return a list of matching chunks.
        Each result contains content, title, source, and score.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
        hits = []
        for i in range(len(results["documents"][0])):
            hits.append(
                {
                    "content": results["documents"][0][i],
                    "title": results["metadatas"][0][i].get("title", ""),
                    "source": results["metadatas"][0][i].get("source", ""),
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                }
            )
        return hits

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
