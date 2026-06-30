from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    title: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, str]]:
    """
    Split the input text into chunks while preserving metadata.

    Returns a list of dictionaries with keys: content, title, source.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    return [
        {"content": chunk, "title": title, "source": source}
        for chunk in chunks
    ]
