import os
from dotenv import load_dotenv

load_dotenv()

# ChromaDB configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chromadb")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_collection")

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# System prompt for the agent
SYSTEM_PROMPT = """
You are an AI assistant that helps users search and add documents to a knowledge base.
When a user asks a question, you should decide whether to search the knowledge base or add new content.
Use the provided tools:
- search_knowledge_base: to perform semantic search.
- add_to_knowledge_base: to add new documents.

Respond in a helpful manner. If you need to call a tool, output the tool name and arguments in JSON format as specified by the tool schema.
"""
