# RAG Agent Project

This repository contains a simple Retrieval-Augmented Generation (RAG) agent built with LangChain, Qdrant, and Ollama. It allows you to add text documents to a vector store and query them using a conversational interface.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Qdrant is running locally (default URL: http://localhost:6333)
# You can use Docker:
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

- **Add documents**: `python -m cli add path/to/file.txt`
- **Search**: `python -m cli search "your query"`
- **Run interactive agent**: `python main.py`

## Configuration

Environment variables (in `.env`):

- `QDRANT_URL`: URL of the Qdrant instance.
- `QDRANT_API_KEY`: API key for Qdrant (if required).
- `QDRANT_COLLECTION`: Name of the collection to use.

Feel free to extend the agent with more tools or integrate it into your own application.
