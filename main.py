from ghost.agent import chat

def main():
    print("Ghost is running. Type 'exit' to quit.\n")
    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        reply, history = chat(history, user_input)
        print(f"\nGhost: {reply}\n")


if __name__ == "__main__":
    main()
