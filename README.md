# RAG Agent

A lightweight Retrieval-Augmented Generation (RAG) agent built with LangChain, Qdrant, and Ollama.  
It allows you to:

- **Add** text documents to a vector store.
- **Search** the knowledge base for relevant passages.
- **Interact** with the agent via a simple CLI or programmatically.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create a `.env` file (or set environment variables)
# Example:
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=
# QDRANT_COLLECTION=rag_collection
```

## Usage

### CLI

```bash
# Add a file
python -m cli add path/to/file.txt

# Search
python -m cli search "your query here"
```

### Agent

```python
from agent import create_agent_executor

agent = create_agent_executor()
response = agent.run("What is the capital of France?")
print(response)
```

## License

MIT
