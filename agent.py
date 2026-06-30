from typing import List

from langchain.agents import create_agent
from langchain_ollama import Ollama
from langchain.schema import AgentAction, AgentFinish

from tools import search_knowledge_base, add_to_knowledge_base
from config import SYSTEM_PROMPT

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


def create_agent_executor() -> AgentExecutor:
    """
    Create and return a LangChain agent that can use the defined tools.
    """
    llm = Ollama(model="llama3")

    agent = create_agent(
        llm=llm,
        tools=TOOLS,
        agent_type="openai-functions",
        verbose=True,
        system_message=SYSTEM_PROMPT,
    )
    return agent
