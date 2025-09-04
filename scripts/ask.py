import argparse
from dotenv import load_dotenv

from rss_mcp.graph_rag_agent import GraphRAGAgent


def main():
    """
    Main function to run the RAG agent query.
    """
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ask a question to the knowledge graph."
    )
    parser.add_argument("question", type=str, help="The question to ask.")
    args = parser.parse_args()

    print("--- Initializing RAG Agent ---")
    try:
        agent = GraphRAGAgent()
        print("Agent initialized.")
    except Exception as e:
        print("\n--- ERROR ---")
        print("Failed to initialize the RAG agent.")
        print(
            "Please ensure your Neo4j database is running and credentials are correct."
        )
        print("You can start the database using: docker-compose up")
        print("Verify your .env file or environment variables.")
        print(f"Error details: {e}")
        return

    print("\n--- Sending Query ---")
    answer = agent.query(args.question)

    print("\n--- Answer ---")
    print(answer)
    print("\n--- Query Complete ---")


if __name__ == "__main__":
    main()
