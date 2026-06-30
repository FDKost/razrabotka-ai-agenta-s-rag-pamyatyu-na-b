# Replace Qdrant with ChromaDB in RAG Agent

## Requirements
- [high] Implement ChromaDB Vector Store: Create a module that initializes a ChromaDB client, handles adding documents with embeddings from Ollama, and performs semantic search with relevance scoring.
- [high] Update RAG Tools: Modify the @tool-decorated functions to interact with the new ChromaDB store: search_knowledge_base(query, max_results) and add_to_knowledge_base(content, title).
- [normal] Adjust Chunking Logic: Ensure RecursiveCharacterTextSplitter continues to split documents into chunks and store metadata correctly in ChromaDB.
- [high] Recreate Agent with New Store: Update create_agent to use the new ChromaDB-based tools and set an appropriate system prompt guiding the agent to use the knowledge base.
- [normal] Rewrite Initialization Script: Provide a script that loads documents from a directory into the ChromaDB store, using the updated chunking and embedding logic.
- [normal] Update Interactive Client: Modify the CLI or simple interface to work with the new tools and vector store, supporting /add, /search, and /quit commands.
