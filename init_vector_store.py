"""
Script to load all documents from a specified directory into the vector store.
Also provides a factory to obtain the configured vector store.
"""

import argparse
from pathlib import Path

from config import VECTOR_STORE, QDRANT_URL, QDRANT_PORT, QDRANT_API_KEY, OLLAMA_MODEL
from qdrant_store import QdrantVectorStore
from chroma_store import ChromaVectorStore
from init_loader import load_directory

def get_vector_store():
    """
    Factory function to return the configured vector store instance.
    """
    if VECTOR_STORE.lower() == "chromadb":
        return ChromaVectorStore()
    elif VECTOR_STORE.lower() == "qdrant":
        return QdrantVectorStore()
    else:
        raise ValueError(f"Unsupported VECTOR_STORE: {VECTOR_STORE}")

def main():
    parser = argparse.ArgumentParser(description="Initialize the vector store with documents.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing .txt files to load.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing collection before loading.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.is_dir():
        raise ValueError(f"Data directory '{data_path}' does not exist.")

    if args.overwrite:
        print("Deleting existing collection...")
        store = get_vector_store()
        store.delete_collection()
        print("Collection deleted. Recreating...")

    load_directory(str(data_path), overwrite=args.overwrite)
    print("Vector store initialization complete.")

if __name__ == "__main__":
    main()
