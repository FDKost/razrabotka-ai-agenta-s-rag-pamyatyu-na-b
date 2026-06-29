from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from src.tools import search_knowledge_base, add_to_knowledge_base
from config import OLLAMA_HOST, OLLAMA_MODEL

llm = Ollama(
    base_url=OLLAMA_HOST,
    model=OLLAMA_MODEL,
    request_timeout=60,
)

tools = [search_knowledge_base, add_to_knowledge_base]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    system_message="You are an AI assistant with RAG memory. Use the provided tools to answer queries.",
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
