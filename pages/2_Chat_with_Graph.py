import streamlit as st
from dotenv import load_dotenv
import os

# Add src to the Python path to allow importing rss_mcp
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from rss_mcp.graph_rag_agent import GraphRAGAgent

# --- Streamlit Page ---
st.set_page_config(page_title="Chat with Graph", page_icon="💬")
st.title("💬 Chat with your Knowledge Graph")

st.markdown(
    """
    Ask questions in natural language about the documents you've ingested.
    The agent will query the knowledge graph to find the answer.
    """
)

# Load environment variables from .env file
load_dotenv()

# Initialize the agent
# We cache the agent in the session state to avoid re-initializing it on every interaction.
if "rag_agent" not in st.session_state:
    with st.spinner("Initializing RAG Agent... This may take a moment."):
        try:
            st.session_state.rag_agent = GraphRAGAgent()
            st.success("Agent initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize RAG Agent: {e}")
            st.warning(
                "Could not connect to Neo4j. Please ensure the database is running "
                "and that you have ingested at least one document."
            )
            st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_agent.query(prompt)
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
