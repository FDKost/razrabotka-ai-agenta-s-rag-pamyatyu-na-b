# RAG Agent with Qdrant and Ollama

This project implements a Retrieval-Augmented Generation (RAG) agent that uses **Qdrant** as a vector store and **Ollama** for embeddings and LLM inference. The agent can ingest documents, search the knowledge base, and answer queries using the stored context.

## Features

- **Qdrant integration** – vector store for embeddings.
- **Ollama embeddings & LLM** – lightweight local inference.
- **Recursive chunking** – splits documents into overlapping chunks.
- **LangChain tools** – `search_knowledge_base` and `add_to_knowledge_base`.
- **Interactive CLI** – `/add`, `/search`, `/quit` commands.
- **RAG agent** – uses tools to answer user queries.

## Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/rag-agent-qdrant-ollama.git
   cd rag-agent-qdrant-ollama
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Copy `.env.example` to `.env` and adjust values:

   ```bash
   cp .env.example .env
   ```

   - `QDRANT_URL`: URL of your Qdrant instance (default `http://localhost:6333`).
   - `QDRANT_API_KEY`: API key for Qdrant (leave empty if not required).
   - `QDRANT_COLLECTION_NAME`: Name of the Qdrant collection.
   - `OLLAMA_HOST`: Ollama host (default `http://localhost:11434`).
   - `OLLAMA_MODEL`: Ollama model name (default `llama3.1`).

5. **Start Ollama**

   Ensure Ollama is running locally or accessible remotely.

6. **Start Qdrant**

   Ensure Qdrant is running locally or accessible remotely. For a local Docker setup:

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

7. **Initialize the database**

   ```bash
   python scripts/init_db.py path/to/your/documents
   ```

## Usage

### Interactive REPL

```bash
python src/cli.py repl
```

Commands:

- `/add <file_path>` – Add a single file to the knowledge base.
- `/search <query>` – Search the knowledge base.
- `/quit` – Exit the REPL.
- Any other input is treated as a chat message and forwarded to the agent.

### Direct Commands

```bash
# Add a file
python src/cli.py add path/to/file.txt

# Search
python src/cli.py search "What is the capital of France?"

# Ingest a directory
python src/cli.py ingest path/to/docs
```

## Project Structure

```
.
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── chunker.py
│   ├── ingestion.py
│   ├── qdrant_client.py
│   ├── tools.py
│   ├── agent.py
│   └── cli.py
├── scripts
│   └── init_db.py
├── .env.example
├── requirements.txt
└── README.md
```

## License

MIT License
