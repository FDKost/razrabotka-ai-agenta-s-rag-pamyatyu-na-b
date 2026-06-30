import os
from pathlib import Path
import logging
from .qdrant_vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO)

vector_store = QdrantVectorStore()

def ingest_directory(directory: Path, overwrite: bool = False):
    """
    Recursively ingest all text files in the given directory.
    """
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith((".txt", ".md", ".pdf")):
                file_path = Path(root) / file_name
                try:
                    if file_name.lower().endswith(".pdf"):
                        from PyPDF2 import PdfReader
                        reader = PdfReader(file_path)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text() or ""
                    else:
                        content = file_path.read_text(encoding="utf-8")
                    title = file_path.stem
                    vector_store.add_documents_with_embeddings(content, title, source=str(file_path))
                    logging.info(f"Ingested {file_path}")
                except Exception as e:
                    logging.error(f"Failed to ingest {file_path}: {e}")
