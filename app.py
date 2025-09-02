import streamlit as st

st.set_page_config(
    page_title="RSS-MCP: Knowledge Graph RAG System",
    page_icon="🧠",
)

st.title("🧠 RSS-MCP: Knowledge Graph RAG System")

st.markdown(
    """
    Welcome to the RSS-MCP (Retrieval-Augmented Generation System with Modular Component Pipeline).
    This application allows you to build and interact with a knowledge graph from your documents.

    ### How to Use This App
    1.  **Ingest Document**: Navigate to the `Ingest Document` page from the sidebar to upload a Markdown file. The system will process it and store it in the Neo4j knowledge graph.
    2.  **Chat with Graph**: Once documents are ingested, go to the `Chat with Graph` page to ask questions in natural language about the content of your documents.

    ### Project Details
    - **Backend**: Python, LangChain, Neo4j
    - **Frontend**: Streamlit
    - **Source Code**: [GitHub Repository](https://github.com/chottokun/MochiRAG) (Note: This is a reference, not the actual repo of this instance).
    """
)
