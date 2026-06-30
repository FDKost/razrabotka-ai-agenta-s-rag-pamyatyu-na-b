# Add @tool decorators to RAG agent tools

## Requirements
- [high] Decorate search_knowledge_base: Import the @tool decorator from langchain.tools and apply it to the search_knowledge_base function, ensuring it returns a dictionary with the search results and optional metadata.
- [high] Decorate add_to_knowledge_base: Import the @tool decorator from langchain.tools and apply it to the add_to_knowledge_base function, ensuring it returns a confirmation message or status after adding a document to the vector store.
