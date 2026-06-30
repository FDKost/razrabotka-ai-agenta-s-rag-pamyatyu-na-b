from agent import create_agent

def main():
    agent = create_agent()
    print("RAG Agent is ready. Type your messages below. Use '/quit' to exit.")
    while True:
        try:
            user_input = input(">> ")
            if user_input.strip().lower() == "/quit":
                print("Exiting.")
                break
            response = agent.run(user_input)
            print(response)
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
