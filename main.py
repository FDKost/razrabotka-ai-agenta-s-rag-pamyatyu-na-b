from agent import create_agent_executor


def main():
    agent = create_agent_executor()
    print("RAG Agent ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("\nUser: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        response = agent.run(user_input)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
