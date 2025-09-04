import pytest
from unittest.mock import MagicMock
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document

# This import will fail initially
from rss_mcp.graph_store import GraphStore
from rss_mcp.config_manager import Neo4jConfig


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks ConfigManager, Neo4jGraph, and EmbeddingManager."""
    # Mock ConfigManager to return a specific Neo4j config
    mock_neo4j_config = Neo4jConfig(
        uri="bolt://mock_uri:7687", username="mock_user", password="mock_password"
    )
    mock_config_manager_instance = MagicMock()
    mock_config_manager_instance.get_neo4j_config.return_value = mock_neo4j_config
    mocker.patch(
        "rss_mcp.graph_store.ConfigManager", return_value=mock_config_manager_instance
    )

    # Mock EmbeddingManager
    mock_embedding_model = MagicMock()
    mock_embedding_manager_instance = MagicMock()
    mock_embedding_manager_instance.get_model.return_value = mock_embedding_model
    mocker.patch(
        "rss_mcp.graph_store.EmbeddingManager",
        return_value=mock_embedding_manager_instance,
    )

    # Mock the Neo4jGraph class
    mock_neo4j_graph_instance = MagicMock()
    mock_neo4j_graph_class = mocker.patch(
        "rss_mcp.graph_store.Neo4jGraph", return_value=mock_neo4j_graph_instance
    )

    return {
        "config_manager_instance": mock_config_manager_instance,
        "embedding_manager_instance": mock_embedding_manager_instance,
        "embedding_model": mock_embedding_model,
        "neo4j_graph_class": mock_neo4j_graph_class,
        "neo4j_graph_instance": mock_neo4j_graph_instance,
        "neo4j_config": mock_neo4j_config,
    }


def test_graph_store_initialization(mock_dependencies):
    """Tests that GraphStore initializes Neo4jGraph correctly."""
    # Act: Initializing GraphStore should trigger the setup
    GraphStore()

    # Assert: Check that Neo4jGraph was initialized with config values
    mock_dependencies["neo4j_graph_class"].assert_called_once_with(
        url=mock_dependencies["neo4j_config"].uri,
        username=mock_dependencies["neo4j_config"].username,
        password=mock_dependencies["neo4j_config"].password,
    )


def test_save_graph(mock_dependencies):
    """Tests that the save_graph method calls the underlying Neo4jGraph method."""
    # Arrange
    graph_store = GraphStore()
    graph_docs = [
        GraphDocument(
            nodes=[], relationships=[], source=Document(page_content="doc 1")
        ),
        GraphDocument(
            nodes=[], relationships=[], source=Document(page_content="doc 2")
        ),
    ]

    # Act
    graph_store.save_graph(graph_docs)

    # Assert
    mock_dependencies[
        "neo4j_graph_instance"
    ].add_graph_documents.assert_called_once_with(graph_docs, include_source=True)


def test_create_vector_index(mocker, mock_dependencies):
    """Tests that the create_vector_index method correctly uses Neo4jVector."""
    # Arrange
    # This time we mock the Neo4jVector class from the module where it's used
    mock_neo4j_vector_class = mocker.patch("rss_mcp.graph_store.Neo4jVector")
    graph_store = GraphStore()

    # Act
    graph_store.create_vector_index()

    # Assert
    # Check that the embedding model was retrieved
    mock_dependencies["embedding_manager_instance"].get_model.assert_called_once()

    # Check that Neo4jVector.from_existing_graph was called correctly
    mock_neo4j_vector_class.from_existing_graph.assert_called_once_with(
        embedding=mock_dependencies["embedding_model"],
        url=mock_dependencies["neo4j_config"].uri,
        username=mock_dependencies["neo4j_config"].username,
        password=mock_dependencies["neo4j_config"].password,
        index_name="chunk_embeddings",
        node_label="Chunk",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )


def test_get_graph_for_visualization(mock_dependencies):
    """Tests that the graph store can fetch and format data for visualization."""
    # Arrange
    graph_store = GraphStore()
    mock_graph = mock_dependencies["neo4j_graph_instance"]

    # Sample raw data returned by graph.query. It's a single row/dict.
    mock_query_result = [
        {
            "nodes": [
                {"id": "n1", "labels": ["Person"], "properties": {"name": "Jules"}},
                {"id": "n2", "labels": ["Project"], "properties": {"name": "rss_mcp"}},
            ],
            "relationships": [
                {
                    "source_id": "n1",
                    "target_id": "n2",
                    "type": "WORKS_ON",
                    "properties": {},
                }
            ],
        }
    ]
    mock_graph.query.return_value = mock_query_result

    # Act
    nodes, edges = graph_store.get_graph_for_visualization()

    # Assert
    mock_graph.query.assert_called_once()

    # Check nodes formatting
    assert len(nodes) == 2
    assert nodes[0]["id"] == "n1"
    assert nodes[0]["label"] == "Jules"
    assert nodes[0]["title"] == "Labels: Person\nname: Jules"
    assert nodes[1]["id"] == "n2"

    # Check edges formatting
    assert len(edges) == 1
    assert edges[0]["source"] == "n1"
    assert edges[0]["to"] == "n2"
    assert edges[0]["label"] == "WORKS_ON"
