from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    title: str,
    source: str,
) -> List[Dict[str, str]]:
    """
    Split the input text into chunks while preserving metadata.

    Returns a list of dictionaries with keys: content, title, source.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    return [
        {"content": chunk, "title": title, "source": source}
        for chunk in chunks
    ]
