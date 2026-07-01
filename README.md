# Агент с RAG‑памятью на Qdrant и Ollama

## Requirements
- [high] Replace vector store with Qdrant: Remove all Chroma usage and implement QdrantVectorStore with client initialization, collection creation, document addition with Ollama embeddings, and semantic search with relevance metric.
- [high] Add Ollama dependency: Ensure pip install includes "langchain-ollama" and use Ollama embeddings and LLM in the project.
- [high] Create RAG tools: Implement @tool decorated functions: search_knowledge_base(query, max_results) and add_to_knowledge_base(content, title).
- [normal] Implement chunking: Use RecursiveCharacterTextSplitter to split documents into chunks before adding to Qdrant, preserving metadata.
- [high] Create agent with RAG integration: Define create_agent function that sets system prompt to use the RAG tools and integrates with LangChain agent framework.
- [normal] Initialization script: Provide a script to load all documents from a specified directory into the Qdrant store using the chunking and embedding pipeline.
- [normal] Interactive CLI client: Implement a simple command-line interface supporting /add, /search, /quit commands to demonstrate agent functionality.
