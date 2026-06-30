import os
from pathlib import Path
from .chunker import chunk_text
from .embeddings import get_ollama_embeddings
from .qdrant_client import QdrantVectorStore

def ingest_directory(directory: Path, overwrite=False):
    """
    Ingests all supported files from a directory into the Qdrant vector store.
    Supported file types: .txt, .md, .pdf
    """
    store = QdrantVectorStore()
    if overwrite:
        # Delete existing collection and recreate
        store.client.delete_collection(collection_name=store.collection_name)
        store._ensure_collection()

    embeddings_model = get_ollama_embeddings()

    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".pdf"}:
            try:
                if file_path.suffix.lower() == ".pdf":
                    from PyPDF2 import PdfReader
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                else:
                    text = file_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                continue

            chunks = chunk_text(text, {"source": str(file_path)})
            contents = [c["content"] for c in chunks]
            metas = [c["metadata"] for c in chunks]
            embeddings = embeddings_model.embed_documents(contents)
            ids = [f"{file_path.stem}_{c['metadata']['chunk_id']}" for c in chunks]
            store.add(embeddings, contents, ids=ids, metadata=metas)
