import pytest
from unittest.mock import MagicMock

# This import will fail initially
from rss_mcp.graph_rag_agent import GraphRAGAgent


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks the core dependencies for the GraphRAGAgent."""
    # Mock LLM and its manager
    mock_llm = MagicMock()
    mock_llm_manager_instance = MagicMock()
    mock_llm_manager_instance.get_llm.return_value = mock_llm
    mocker.patch(
        "rss_mcp.graph_rag_agent.LLMManager", return_value=mock_llm_manager_instance
    )

    # Mock Neo4jGraph and GraphStore
    mock_neo4j_graph = MagicMock()
    mock_graph_store_instance = MagicMock()
    mock_graph_store_instance.graph = mock_neo4j_graph
    mocker.patch(
        "rss_mcp.graph_rag_agent.GraphStore", return_value=mock_graph_store_instance
    )

    # Mock the QA Chain
    mock_qa_chain_instance = MagicMock()
    mock_qa_chain_instance.invoke.return_value = {
        "result": "Jules is a skilled software engineer."
    }
    mock_qa_chain_class = mocker.patch(
        "rss_mcp.graph_rag_agent.GraphCypherQAChain.from_llm",
        return_value=mock_qa_chain_instance,
    )

    return {
        "llm": mock_llm,
        "neo4j_graph": mock_neo4j_graph,
        "qa_chain_class": mock_qa_chain_class,
        "qa_chain_instance": mock_qa_chain_instance,
    }


def test_agent_initialization(mock_dependencies):
    """Tests that the agent initializes the GraphCypherQAChain correctly."""
    GraphRAGAgent()

    # Assert that the chain's factory method was called with the correct graph and llm
    mock_dependencies["qa_chain_class"].assert_called_once_with(
        graph=mock_dependencies["neo4j_graph"],
        llm=mock_dependencies["llm"],
        verbose=True,
    )


def test_agent_query_method(mock_dependencies):
    """Tests that the query method correctly calls the QA chain."""
    agent = GraphRAGAgent()
    question = "Who is Jules?"

    # Act
    result = agent.query(question)

    # Assert that the chain's invoke method was called with the question
    mock_dependencies["qa_chain_instance"].invoke.assert_called_once_with(
        {"query": question}
    )

    # Assert that the result from the agent is the one from the chain
    assert result == "Jules is a skilled software engineer."
