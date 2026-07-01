import os
from typing import List, Dict, Any

from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document

from config import CHROMA_PATH, CHROMA_COLLECTION, OLLAMA_MODEL
from chunker import chunk_document

class ChromaVectorStoreWrapper:
    """
    Wrapper around langchain's Chroma vector store to provide a simple API
    for adding documents and searching.
    """

    def __init__(self, collection_name: str = CHROMA_COLLECTION, persist_directory: str = CHROMA_PATH):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = OllamaEmbeddings(model=OLLAMA_MODEL)
        # Initialize or load existing collection
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
        )

    def add_document(self, text: str, title: str, source: str) -> int:
        """
        Chunk the document, embed, and upsert into Chroma.
        Returns the number of chunks added.
        """
        chunks = chunk_document(text, title, source)
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "content": chunk["text"],
                },
            )
            documents.append(doc)
        self.vector_store.add_documents(documents)
        return len(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic search and return a list of matching chunks.
        Each result contains content, title, source, and score.
        """
        results = self.vector_store.similarity_search_with_score(query, k)
        hits = []
        for doc, score in results:
            metadata = doc.metadata or {}
            hits.append(
                {
                    "content": doc.page_content,
                    "title": metadata.get("title", ""),
                    "source": metadata.get("source", ""),
                    "score": score,
                }
            )
        return hits

    def delete_collection(self):
        """Delete the entire collection."""
        self.vector_store.delete_collection()

    def persist(self):
        """Persist the collection to disk."""
        self.vector_store.persist()

def get_vector_store() -> ChromaVectorStoreWrapper:
    """
    Factory function to return a singleton ChromaVectorStoreWrapper instance.
    """
    if not hasattr(get_vector_store, "_instance"):
        get_vector_store._instance = ChromaVectorStoreWrapper()
    return get_vector_store._instance
