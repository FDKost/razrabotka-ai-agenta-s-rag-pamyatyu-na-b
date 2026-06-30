from langchain.agents import AgentExecutor, create_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from .tools import search_knowledge_base, add_to_knowledge_base
from .config import OLLAMA_HOST, OLLAMA_MODEL

llm = Ollama(
    base_url=OLLAMA_HOST,
    model=OLLAMA_MODEL,
    request_timeout=60,
)

tools = [search_knowledge_base, add_to_knowledge_base]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_agent(
    llm=llm,
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    system_message="You are an AI assistant with RAG memory. Use the provided tools to search the knowledge base and add documents.",
    verbose=True,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
