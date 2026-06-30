# AI Agent with RAG Memory using Qdrant, Ollama, and LangChain

This project implements a Retrieval-Augmented Generation (RAG) agent that can search and add documents to a vector store backed by Qdrant. It uses Ollama for embeddings and LLM inference, and LangChain for orchestration.

## Features

- **Semantic Search**: Query the knowledge base with a natural language prompt.
- **Document Ingestion**: Add new documents to the knowledge base.
- **Chunking**: Documents are split into manageable chunks with metadata.
- **RAG Agent**: An agent that uses the two tools to answer user queries.
- **Bulk Loader**: Load all text files from a directory into the vector store.
- **CLI Client**: Interactive command-line interface for adding and searching documents.

## Installation

```bash
git clone https://github.com/yourusername/rag-agent.git
cd rag-agent
pip install -r requirements.txt
```

## Usage

```bash
# Load all text files from a directory
python init_loader.py --path ./data --overwrite

# Start the interactive CLI
python cli.py
```

## Environment Variables

Create a `.env` file in the project root:

```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=YOUR_API_KEY
QDRANT_COLLECTION=rag_collection
```

Make sure Qdrant is running and accessible at the URL above.

## License

MIT License
