from langchain_community.embeddings import OllamaEmbeddings
from .config import OLLAMA_HOST, OLLAMA_MODEL

def get_embedding_model():
    return OllamaEmbeddings(
        base_url=OLLAMA_HOST,
        model=OLLAMA_MODEL,
    )
