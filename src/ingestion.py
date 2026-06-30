import uuid
from pathlib import Path
from tqdm import tqdm
from .chunker import chunk_text
from .embeddings import get_embedding_model
from .qdrant_client import upsert, create_collection
from .config import QDRANT_COLLECTION_NAME
import PyPDF2

def _read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")

def _read_pdf_file(file_path: Path) -> str:
    reader = PyPDF2.PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def ingest_directory(directory: Path):
    create_collection()
    for file_path in directory.rglob("*"):
        if file_path.suffix.lower() in {".txt", ".md"}:
            content = _read_text_file(file_path)
        elif file_path.suffix.lower() == ".pdf":
            content = _read_pdf_file(file_path)
        else:
            continue
        add_document_from_content(content, file_path.stem)

def add_document(file_path: Path):
    if file_path.suffix.lower() in {".txt", ".md"}:
        content = _read_text_file(file_path)
    elif file_path.suffix.lower() == ".pdf":
        content = _read_pdf_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    add_document_from_content(content, file_path.stem)

def add_document_from_content(content: str, title: str):
    create_collection()
    chunks = chunk_text(content, {"title": title})
    embeddings = get_embedding_model()
    vectors = [embeddings.embed_query(chunk["content"]) for chunk in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]
    payloads = [chunk["metadata"] for chunk in chunks]
    upsert(QDRANT_COLLECTION_NAME, ids, vectors, payloads)
