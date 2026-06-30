from langchain_ollama.embeddings import OllamaEmbeddings
from src.config import OLLAMA_HOST, OLLAMA_MODEL

def get_ollama_embeddings():
    """
    Returns an instance of OllamaEmbeddings configured with the host and model.
    """
    return OllamaEmbeddings(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
