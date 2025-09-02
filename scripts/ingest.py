import argparse
import os
from dotenv import load_dotenv

# Add src to the Python path to allow importing rss_mcp
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from rss_mcp.document_processor import DocumentProcessor
from rss_mcp.graph_converter import GraphConverter
from rss_mcp.graph_store import GraphStore


def main():
    """
    Main function to run the data ingestion pipeline.
    """
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ingest a document into the knowledge graph."
    )
    parser.add_argument(
        "file_path", type=str, help="The path to the document file to ingest."
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        return

    print("--- Starting Ingestion Pipeline ---")

    # 1. Process Document
    print(f"\n[Step 1/3] Processing document: {args.file_path}")
    doc_processor = DocumentProcessor()
    chunks = doc_processor.process_file(args.file_path)
    print(f"Successfully created {len(chunks)} chunks.")

    # 2. Convert to Graph
    print("\n[Step 2/3] Converting chunks to graph documents...")
    graph_converter = GraphConverter()
    graph_documents = graph_converter.convert_to_graph(chunks)
    print(f"Successfully converted chunks to {len(graph_documents)} graph documents.")

    # 3. Store in Neo4j
    print("\n[Step 3/4] Storing graph in Neo4j...")
    try:
        graph_store = GraphStore()
        graph_store.save_graph(graph_documents)
        print("Successfully stored graph in Neo4j.")

        # 4. Create Vector Index
        print("\n[Step 4/4] Creating vector index...")
        graph_store.create_vector_index()

    except Exception as e:
        print("\n--- ERROR ---")
        print("Failed to connect to or store data in Neo4j.")
        print(
            "Please ensure your Neo4j database is running and credentials are correct."
        )
        print("You can start the database using: docker-compose up")
        print("Verify your .env file or environment variables (NEO4J_URI, etc.).")
        print(f"Error details: {e}")
        return

    print("\n--- Ingestion Pipeline Complete ---")


if __name__ == "__main__":
    main()
