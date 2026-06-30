# RAG Agent Project

This repository contains a simple Retrieval-Augmented Generation (RAG) agent built with LangChain, Ollama, and Qdrant.  
It allows you to:

* **Add** text documents to a vector store.
* **Search** the knowledge base via semantic search.
* Interact with the agent through a command‑line interface or programmatically.

## Setup

```bash
# Clone the repo
git clone https://github.com/your-username/rag-agent.git
cd rag-agent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root (a template is provided) and set:

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # leave empty if no key
QDRANT_COLLECTION=rag_collection
```

Ensure a Qdrant instance is running at the specified URL.

## Usage

### Load documents

```bash
python init_loader.py --path /path/to/text/files
```

### CLI

```bash
python cli.py add /path/to/file.txt
python cli.py search "your query"
```

### Interactive Agent

```bash
python main.py
```

Type your questions and the agent will decide whether to search or add content.  
Type `/quit` to exit.

## License

MIT License
