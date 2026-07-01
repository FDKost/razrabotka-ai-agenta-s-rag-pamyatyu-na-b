import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")

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
        "QDRANT_HOST": QDRANT_HOST,
        "QDRANT_PORT": QDRANT_PORT,
        "QDRANT_COLLECTION": QDRANT_COLLECTION,
        "OLLAMA_HOST": OLLAMA_HOST,
        "OLLAMA_PORT": OLLAMA_PORT,
        "OLLAMA_EMBED_MODEL": OLLAMA_EMBED_MODEL,
        "OLLAMA_LLM_MODEL": OLLAMA_LLM_MODEL,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
    }
