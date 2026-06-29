from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)

def chunk_text(text: str, metadata: dict):
    """
    Returns list of dicts:
        content: str
        metadata: dict
    """
    chunks = splitter.split_text(text)
    result = []
    for idx, chunk in enumerate(chunks):
        result.append(
            {
                "content": chunk,
                "metadata": {**metadata, "chunk_id": idx},
            }
        )
    return result
