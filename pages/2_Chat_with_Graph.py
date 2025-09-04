import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from dotenv import load_dotenv

from rss_mcp.graph_rag_agent import GraphRAGAgent


# --- Helper Function ---
def format_context_for_visualization(context):
    """Formats raw graph data into lists of agraph Nodes and Edges."""
    nodes = []
    edges = []
    if not isinstance(context, list) or not context:
        return nodes, edges

    # The context is often a list of dicts from the graph query
    graph_data = context[0]

    # Process nodes
    for node_info in graph_data.get("nodes", []):
        node_id = node_info.get("id", "")
        node_label = str(node_info.get("properties", {}).get("name", node_id))
        nodes.append(Node(id=node_id, label=node_label, size=25))

    # Process relationships
    for edge_info in graph_data.get("relationships", []):
        source = edge_info.get("source_id")
        target = edge_info.get("target_id")
        label = edge_info.get("type", "")
        if source and target:
            edges.append(Edge(source=source, target=target, label=label))

    return nodes, edges


# --- Streamlit Page ---
st.set_page_config(page_title="Chat with Graph", page_icon="💬")
st.title("💬 Chat with your Knowledge Graph")

st.markdown(
    """
    Ask questions in natural language about the documents you've ingested.
    The agent will query the knowledge graph to find the answer and show you the data it used.
    """
)

# Load environment variables
load_dotenv()

# Initialize agent
if "rag_agent" not in st.session_state:
    with st.spinner("Initializing RAG Agent..."):
        try:
            st.session_state.rag_agent = GraphRAGAgent()
            st.success("Agent initialized!")
        except Exception as e:
            st.error(f"Failed to initialize RAG Agent: {e}")
            st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context_graph" in message:
            with st.expander("Show Answer Context"):
                agraph(
                    nodes=message["context_graph"]["nodes"],
                    edges=message["context_graph"]["edges"],
                    config=Config(
                        height=300,
                        width=700,
                        directed=True,
                        physics=True,
                        hierarchical=False,
                    ),
                )

# Accept user input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_agent.query(prompt)
            answer = response.get("answer", "Sorry, I couldn't find an answer.")
            context = response.get("context")

            st.markdown(answer)

            message_to_save = {"role": "assistant", "content": answer}

            if context:
                nodes, edges = format_context_for_visualization(context)
                if nodes:
                    with st.expander("Show Answer Context"):
                        agraph(
                            nodes=nodes,
                            edges=edges,
                            config=Config(
                                height=300,
                                width=700,
                                directed=True,
                                physics=True,
                                hierarchical=False,
                            ),
                        )
                    # Save the graph data for re-rendering
                    message_to_save["context_graph"] = {"nodes": nodes, "edges": edges}

    st.session_state.messages.append(message_to_save)
