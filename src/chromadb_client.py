from chromadb import Client
from chromadb.config import Settings
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")

client = Client(
    settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chromadb",
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )
)

def create_collection():
    """Create or reset the collection."""
    if COLLECTION_NAME in client.list_collections():
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def upsert_vectors(vectors: List[Dict]):
    """Upsert a list of vectors into the collection."""
    ids = [v["id"] for v in vectors]
    embeddings = [v["vector"] for v in vectors]
    payloads = [v["payload"] for v in vectors]
    collection = client.get_collection(COLLECTION_NAME)
    collection.upsert(vectors=ids, embeddings=embeddings, metadatas=payloads)

def similarity_search(query_vector: List[float], k: int):
    """Perform a semantic similarity search."""
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["embeddings", "metadatas", "distances"],
    )
    hits = []
    for idx in range(len(results["ids"][0])):
        hit = {
            "id": results["ids"][0][idx],
            "score": 1 - results["distances"][0][idx],  # convert distance to similarity
            "metadata": results["metadatas"][0][idx],
            "embedding": results["embeddings"][0][idx],
        }
        hits.append(hit)
    return hits
