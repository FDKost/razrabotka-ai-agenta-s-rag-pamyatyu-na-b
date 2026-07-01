from typing import List

from langchain.agents import create_agent
from langchain_ollama import Ollama
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool

from tools import search_knowledge_base, add_to_knowledge_base
from config import SYSTEM_PROMPT, OLLAMA_LLM_MODEL


# Define the tools
TOOLS: List[Tool] = [
    search_knowledge_base,
    add_to_knowledge_base,
]


def create_agent_executor():
    """
    Create and return a LangChain agent that can use the defined tools.
    """
    llm = Ollama(model=OLLAMA_LLM_MODEL)

    agent = create_agent(
        llm=llm,
        tools=TOOLS,
        agent_type="openai-functions",
        verbose=True,
        system_message=SYSTEM_PROMPT,
    )
    return agent
