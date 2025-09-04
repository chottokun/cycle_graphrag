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

    def get_graph_for_visualization(self):
        """
        Fetches all nodes and relationships from the graph and formats them
        for visualization with pyvis.
        """
        print("Fetching graph data for visualization...")
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN
            collect(DISTINCT {
                id: elementId(n),
                labels: labels(n),
                properties: properties(n)
            }) AS nodes,
            collect(DISTINCT {
                source_id: elementId(startNode(r)),
                target_id: elementId(endNode(r)),
                type: type(r),
                properties: properties(r)
            }) AS relationships
        """
        raw_result = self.graph.query(query)

        # The query returns a list with one element which is a dict
        if not raw_result or not raw_result[0]:
            return [], []

        data = raw_result[0]
        nodes = []
        edges = []

        for node_data in data.get("nodes", []):
            # Use 'name' or first property as label, fallback to ID
            label = node_data["properties"].get("name", node_data["id"])
            title = f"Labels: {', '.join(node_data['labels'])}\n"
            title += "\n".join(
                [f"{k}: {v}" for k, v in node_data["properties"].items()]
            )

            nodes.append(
                {
                    "id": node_data["id"],
                    "label": str(label),
                    "title": title,
                    "group": node_data["labels"][0]
                    if node_data["labels"]
                    else "Unknown",
                }
            )

        for edge_data in data.get("relationships", []):
            if edge_data.get("source_id") and edge_data.get("target_id"):
                edges.append(
                    {
                        "source": edge_data["source_id"],
                        "to": edge_data["target_id"],
                        "label": edge_data["type"],
                    }
                )

        print(f"Fetched {len(nodes)} nodes and {len(edges)} edges.")
        return nodes, edges
