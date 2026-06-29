from langchain_community.embeddings import OllamaEmbeddings
from config import OLLAMA_HOST, OLLAMA_MODEL

embedding_model = OllamaEmbeddings(
    base_url=OLLAMA_HOST,
    model=OLLAMA_MODEL,
    request_timeout=60,
)

def get_embedding(text: str):
    return embedding_model.embed_query(text)
