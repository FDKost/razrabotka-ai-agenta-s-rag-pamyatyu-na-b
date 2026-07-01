import uuid
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_document(text: str, title: str, source: str) -> List[Dict]:
    """
    Split the input text into chunks while preserving metadata.

    Returns a list of dictionaries with keys: id, text, title, source.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    return [
        {
            "id": str(uuid.uuid4()),
            "text": chunk,
            "title": title,
            "source": source,
        }
        for chunk in chunks
    ]
