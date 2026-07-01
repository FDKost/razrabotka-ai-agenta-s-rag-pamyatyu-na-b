from init_vector_store import get_vector_store
from agent import create_agent_executor

def main():
    # Initialize vector store (singleton)
    store = get_vector_store()
    # Create agent
    agent = create_agent_executor()
    print("Agent initialized. Ready to use.")

if __name__ == "__main__":
    main()
