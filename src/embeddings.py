from langchain_ollama import OllamaEmbeddings
from .config import OLLAMA_HOST, OLLAMA_MODEL

# Initialize the Ollama embeddings model
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_HOST,
    model=OLLAMA_MODEL,
)

def embed(text: str):
    """
    Return the embedding vector for a given text string.
    """
    return embeddings.embed_query(text)
