from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.vectorstores import Neo4jVector

from .config_manager import ConfigManager
from .embedding_manager import EmbeddingManager


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
        self.neo4j_config = config_manager.get_neo4j_config()

        self.graph = Neo4jGraph(
            url=self.neo4j_config.uri,
            username=self.neo4j_config.username,
            password=self.neo4j_config.password,
        )

    def save_graph(self, graph_documents: List[GraphDocument]):
        """
        Saves a list of graph documents to the Neo4j database.

        Args:
            graph_documents: A list of GraphDocument objects to be stored.
        """
        print(f"Storing {len(graph_documents)} graph documents in Neo4j...")
        # include_source=True is important to link chunks to their source document
        self.graph.add_graph_documents(graph_documents, include_source=True)
        print("Storage complete.")

    def create_vector_index(self):
        """
        Creates a vector index in Neo4j for similarity search on Chunk nodes.
        This will automatically compute embeddings for nodes that don't have them.
        """
        print("Creating Neo4j vector index for Chunk embeddings...")
        embedding_manager = EmbeddingManager()
        embedding_model = embedding_manager.get_model()

        Neo4jVector.from_existing_graph(
            embedding=embedding_model,
            url=self.neo4j_config.uri,
            username=self.neo4j_config.username,
            password=self.neo4j_config.password,
            index_name="chunk_embeddings",
            node_label="Chunk",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
        print("Vector index creation/update complete.")
