# Revision: Implement Qdrant-based RAG Agent with @tool-decorated Tools

## Requirements
- [high] Replace vector store with Qdrant: Implement a QdrantVectorStore module that initializes a Qdrant client, creates/uses a collection, adds documents with embeddings from Ollama, and performs semantic search with relevance metric.
- [high] Decorate tools with @tool: Ensure that both search_knowledge_base and add_to_knowledge_base functions are decorated with langchain.tools.tool decorator and expose correct signatures.
- [normal] Integrate Qdrant store in agent: Update the create_agent function to use the new QdrantVectorStore and include the @tool-decorated functions as tools.
- [normal] Verify chunking and metadata: Confirm that documents are split using RecursiveCharacterTextSplitter before insertion and that metadata (title, source) is stored in Qdrant.
- [normal] Test interactive client: Run the CLI client to add documents, search, and ensure responses are correct with the new Qdrant backend.
