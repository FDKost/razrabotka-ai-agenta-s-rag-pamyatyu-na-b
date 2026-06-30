from langchain_community.embeddings import OllamaEmbeddings
from .config import OLLAMA_HOST, OLLAMA_MODEL

def get_embedding_model():
    """
    Returns an OllamaEmbeddings instance configured with the host and model.
    """
    return OllamaEmbeddings(
        base_url=OLLAMA_HOST,
        model=OLLAMA_MODEL,
    )
