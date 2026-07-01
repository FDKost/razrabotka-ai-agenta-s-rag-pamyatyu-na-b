import os
from dotenv import load_dotenv

load_dotenv()

# ChromaDB configuration
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chromadb")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_collection")

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# System prompt for the agent
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """
You are an AI assistant that helps users search and add documents to a knowledge base.
When a user asks a question, you should decide whether to search the knowledge base or add new content.
Use the provided tools:
- search_knowledge_base: to perform semantic search.
- add_to_knowledge_base: to add new documents.

Respond in a helpful manner. If you need to call a tool, output the tool name and arguments in JSON format as specified by the tool schema.
""",
)

def get_config():
    """Return a dictionary of all configuration values."""
    return {
        "CHROMA_PATH": CHROMA_PATH,
        "CHROMA_COLLECTION": CHROMA_COLLECTION,
        "OLLAMA_HOST": OLLAMA_HOST,
        "OLLAMA_PORT": OLLAMA_PORT,
        "OLLAMA_MODEL": OLLAMA_MODEL,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
    }
