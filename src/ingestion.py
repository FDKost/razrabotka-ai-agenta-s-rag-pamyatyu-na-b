import os
from pathlib import Path
from typing import List
from .chunker import chunk_text
from .embeddings import get_embedding_model
from .qdrant_client import QdrantClientWrapper
from .config import QDRANT_COLLECTION_NAME
import logging
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _read_file(file_path: Path) -> str:
    """
    Reads a file and returns its text content.
    Supports .txt and .pdf files.
    """
    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif file_path.suffix.lower() == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

def ingest_directory(directory: Path, overwrite: bool = False):
    """
    Ingest all supported files from a directory into Qdrant.
    """
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")

    client = QdrantClientWrapper()
    if overwrite:
        client.delete_collection()
        client._ensure_collection()

    embedding_model = get_embedding_model()
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}]
    logger.info(f"Found {len(files)} files to ingest.")

    for file_path in files:
        try:
            content = _read_file(file_path)
        except Exception as e:
            logger.warning(f"Skipping {file_path}: {e}")
            continue

        metadata = {"title": file_path.stem, "source": str(file_path)}
        chunks = chunk_text(content, metadata)
        embeddings = embedding_model.embed_documents([c["content"] for c in chunks])

        vectors = []
        for idx, (chunk, embed) in enumerate(zip(chunks, embeddings)):
            vector = {
                "id": str(uuid.uuid4()),
                "vector": embed,
                "payload": {**chunk["metadata"]},
            }
            vectors.append(vector)

        client.upsert(vectors)
        logger.info(f"Ingested {file_path} with {len(chunks)} chunks.")

    logger.info("Ingestion complete.")
