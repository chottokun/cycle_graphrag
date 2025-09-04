import streamlit as st
from streamlit_agraph import agraph
from pyvis.network import Network
from dotenv import load_dotenv

from rss_mcp.graph_store import GraphStore

# --- Streamlit Page ---
st.set_page_config(page_title="Visualize Knowledge Graph", page_icon="🕸️")
st.title("🕸️ Visualize Knowledge Graph")

st.markdown(
    """
    Explore the connections in your knowledge graph. This is a complete view of all
    entities and relationships currently stored in the Neo4j database.
    """
)

# Load environment variables from .env file
load_dotenv()


@st.cache_data
def fetch_graph_data():
    """
    Fetches graph data and caches it to prevent re-fetching on every interaction.
    """
    try:
        graph_store = GraphStore()
        return graph_store.get_graph_for_visualization()
    except Exception as e:
        st.error(f"Failed to connect to Neo4j and fetch data: {e}")
        st.warning("Please ensure the Neo4j database is running and accessible.")
        return None, None


nodes_data, edges_data = fetch_graph_data()

if nodes_data:
    st.info(
        f"Displaying a graph with {len(nodes_data)} nodes and {len(edges_data)} edges."
    )

    # Create pyvis network object
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        notebook=True,
        cdn_resources="in_line",
    )

    # Add nodes and edges from the fetched data
    for node in nodes_data:
        net.add_node(
            node["id"], label=node["label"], title=node["title"], group=node["group"]
        )

    for edge in edges_data:
        net.add_edge(edge["source"], edge["to"], label=edge["label"])

    # Configure physics for a better layout
    net.set_options("""
    var options = {
      "physics": {
        "repulsion": {
          "centralGravity": 0.2,
          "springLength": 200,
          "springConstant": 0.05,
          "nodeDistance": 150,
          "damping": 0.09
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "repulsion"
      }
    }
    """)

    # Convert pyvis network to a Streamlit agraph component
    agraph(net.nodes, net.edges, net.get_network_options())

else:
    st.warning("No graph data to display. Have you ingested any documents yet?")
