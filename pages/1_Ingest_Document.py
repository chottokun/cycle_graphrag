import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from rss_mcp.document_processor import DocumentProcessor
from rss_mcp.graph_converter import GraphConverter
from rss_mcp.graph_store import GraphStore


# --- Helper Function ---
def ingest_document(file_path: str):
    """
    The complete ingestion pipeline for a single document.
    """
    st.info(f"Starting ingestion for: `{os.path.basename(file_path)}`")

    with st.spinner("Step 1/3: Processing and chunking document..."):
        doc_processor = DocumentProcessor()
        chunks = doc_processor.process_file(file_path)
        st.success(f"Document processed into {len(chunks)} chunks.")

    with st.spinner("Step 2/4: Converting chunks to graph documents..."):
        graph_converter = GraphConverter()
        graph_documents = graph_converter.convert_to_graph(chunks)
        st.success(f"Converted chunks to {len(graph_documents)} graph documents.")

    with st.spinner("Step 3/4: Storing graph in Neo4j..."):
        graph_store = GraphStore()
        graph_store.save_graph(graph_documents)
        st.success("Graph stored successfully in Neo4j!")

    with st.spinner("Step 4/4: Creating vector index for fast search..."):
        graph_store.create_vector_index()
        st.success("Vector index created successfully!")


# --- Streamlit Page ---
st.set_page_config(page_title="Ingest Document", page_icon="📄")
st.title("📄 Ingest Document")

st.markdown(
    """
    Upload a Markdown document to process it and add its contents to the knowledge graph.
    The system will extract entities and relationships and store them in the Neo4j database.
    """
)

# Load environment variables from .env file
load_dotenv()

uploaded_file = st.file_uploader(
    "Choose a Markdown file",
    type="md",
    help="Upload a Markdown (.md) file to be processed.",
)

if uploaded_file is not None:
    # To handle the uploaded file, we need to save it to a temporary location
    # so our existing processing functions can read it by file path.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.info(f"File `{uploaded_file.name}` uploaded successfully. Processing now...")

        # Run the full ingestion pipeline
        ingest_document(tmp_file_path)

        st.balloons()
        st.success("🎉 Ingestion Pipeline Complete! 🎉")

    except Exception as e:
        st.error(f"An error occurred during the ingestion process: {e}")
        # Add specific advice for common errors
        if "Connection refused" in str(e) or "database not available" in str(e).lower():
            st.warning(
                "Could not connect to Neo4j. Please ensure the database is running. "
                "You can start it with `docker-compose up`."
            )
    finally:
        # Clean up the temporary file
        if "tmp_file_path" in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
