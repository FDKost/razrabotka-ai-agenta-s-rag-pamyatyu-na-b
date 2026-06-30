from pathlib import Path
from typing import List
from .chunker import chunk_text
from .qdrant_client import QdrantClientWrapper

client = QdrantClientWrapper()

def ingest_file(file_path: Path):
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")
    metadata = {"title": file_path.stem, "file_path": str(file_path)}
    chunks = chunk_text(text, metadata)
    client.add_documents(chunks)

def ingest_directory(directory: Path):
    for path in directory.rglob("*"):
        if path.is_file():
            ingest_file(path)
