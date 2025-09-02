from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument

from .config_manager import ConfigManager


class GraphStore:
    """
    Manages the connection to and storage of graphs in a Neo4j database.
    """

    def __init__(self):
        """
        Initializes the GraphStore by connecting to the Neo4j database
        using configuration from the ConfigManager.
        """
        config_manager = ConfigManager()
        neo4j_config = config_manager.get_neo4j_config()

        self.graph = Neo4jGraph(
            url=neo4j_config.uri,
            username=neo4j_config.username,
            password=neo4j_config.password,
        )

    def save_graph(self, graph_documents: List[GraphDocument]):
        """
        Saves a list of graph documents to the Neo4j database.

        Args:
            graph_documents: A list of GraphDocument objects to be stored.
        """
        print(f"Storing {len(graph_documents)} graph documents in Neo4j...")
        self.graph.add_graph_documents(graph_documents)
        print("Storage complete.")
