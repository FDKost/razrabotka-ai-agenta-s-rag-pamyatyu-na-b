import os
import uuid
from pathlib import Path
from typing import List, Dict
from .chunker import chunk_text
from .embeddings import get_embedding_model
from .qdrant_client import QdrantVectorStore

vector_store = QdrantVectorStore()
embedding_model = get_embedding_model()

def ingest_file(file_path: Path):
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    metadata = {"source": str(file_path), "title": file_path.stem}
    chunks = chunk_text(content, metadata)
    documents = []
    for chunk in chunks:
        embedding = embedding_model.embed_query(chunk["content"])
        doc_id = str(uuid.uuid4())
        documents.append(
            {
                "id": doc_id,
                "embedding": embedding,
                "metadata": {"content": chunk["content"], **chunk["metadata"]},
            }
        )
    vector_store.add_documents(documents)
    print(f"Ingested {len(chunks)} chunks from {file_path}")

def ingest_directory(directory: Path):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".txt", ".md", ".pdf", ".docx")):
                file_path = Path(root) / file
                ingest_file(file_path)
