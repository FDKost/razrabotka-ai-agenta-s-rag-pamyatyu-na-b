from langchain.tools import tool
from .qdrant_client import QdrantClientWrapper

client = QdrantClientWrapper()

@tool(name="search_knowledge_base", description="Search the knowledge base. Returns top results.")
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    results = client.search(query, limit=max_results)
    if not results:
        return "No results found."
    formatted = []
    for idx, res in enumerate(results, 1):
        content = res["content"][:200]  # truncate for brevity
        metadata = res["metadata"]
        title = metadata.get("title", "Untitled")
        formatted.append(f"{idx}. {title}: {content}...")
    return "\n".join(formatted)

@tool(name="add_to_knowledge_base", description="Add a document to the knowledge base.")
def add_to_knowledge_base(content: str, title: str) -> str:
    doc = {"content": content, "metadata": {"title": title}}
    client.add_documents([doc])
    return f"Document '{title}' added to knowledge base."
