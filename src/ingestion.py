import os
from pathlib import Path
from tqdm import tqdm
from .chunker import chunk_text
from .embeddings import get_embedding
from .qdrant_client import upsert_vectors, create_collection

def read_file(file_path: Path):
    if file_path.suffix.lower() in [".txt"]:
        return file_path.read_text(encoding="utf-8")
    # Add more file type handlers if needed
    raise ValueError(f"Unsupported file type: {file_path.suffix}")

def ingest_directory(directory: Path):
    create_collection()
    all_vectors = []
    for file_path in tqdm(directory.rglob("*"), desc="Ingesting files"):
        if file_path.is_file():
            try:
                text = read_file(file_path)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                continue
            metadata = {"source": str(file_path)}
            chunks = chunk_text(text, metadata)
            for chunk in chunks:
                vector = get_embedding(chunk["content"])
                point_id = f"{file_path.stem}_{chunk['metadata']['chunk_id']}"
                all_vectors.append(
                    {
                        "id": point_id,
                        "vector": vector,
                        "payload": {
                            "content": chunk["content"],
                            "source": metadata["source"],
                            "chunk_id": chunk["metadata"]["chunk_id"],
                        },
                    }
                )
    upsert_vectors(all_vectors)
    print(f"Ingested {len(all_vectors)} vectors.")
