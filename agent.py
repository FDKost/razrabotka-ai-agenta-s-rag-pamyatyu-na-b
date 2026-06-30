from typing import List

from langchain.agents import initialize_agent, AgentExecutor, Tool
from langchain_ollama import Ollama
from langchain.schema import AgentAction, AgentFinish

from tools import search_knowledge_base, add_to_knowledge_base

# Define the tools
TOOLS: List[Tool] = [
    Tool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        description="Use this tool to perform a semantic search on the knowledge base.",
    ),
    Tool.from_function(
        func=add_to_knowledge_base,
        name="add_to_knowledge_base",
        description="Use this tool to add a new document or chunk to the knowledge base.",
    ),
]


def create_agent() -> AgentExecutor:
    """
    Create and return a LangChain agent that can use the defined tools.
    """
    llm = Ollama(model="llama3")

    system_prompt = """
You are an AI assistant that helps users search and add documents to a knowledge base.
When a user asks a question, you should decide whether to search the knowledge base or add new content.
Use the provided tools:
- search_knowledge_base: to perform semantic search.
- add_to_knowledge_base: to add new documents.

Respond in a helpful manner. If you need to call a tool, output the tool name and arguments in JSON format as specified by the tool schema.
"""

    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent="openai-functions",
        verbose=True,
        system_message=system_prompt,
    )
    return agent
