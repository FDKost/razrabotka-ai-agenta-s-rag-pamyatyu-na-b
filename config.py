import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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
