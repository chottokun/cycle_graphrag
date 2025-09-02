import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from rss_mcp.graph_rag_agent import GraphRAGAgent


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks the core dependencies for the Hybrid GraphRAGAgent."""
    # Mock Managers
    mock_llm_output = AIMessage(content="A final combined answer.")
    # We use side_effect to handle both llm() and llm.invoke() calls
    mock_llm = MagicMock(side_effect=lambda *args, **kwargs: mock_llm_output)

    mock_llm_manager_instance = MagicMock()
    mock_llm_manager_instance.get_llm.return_value = mock_llm
    mocker.patch(
        "rss_mcp.graph_rag_agent.LLMManager", return_value=mock_llm_manager_instance
    )

    mock_embedding_model = MagicMock()
    mock_embedding_manager_instance = MagicMock()
    mock_embedding_manager_instance.get_model.return_value = mock_embedding_model
    mocker.patch(
        "rss_mcp.graph_rag_agent.EmbeddingManager",
        return_value=mock_embedding_manager_instance,
    )

    # Mock Neo4jGraph and GraphStore
    mock_neo4j_graph = MagicMock()
    mock_graph_store_instance = MagicMock()
    mock_graph_store_instance.graph = mock_neo4j_graph
    mocker.patch(
        "rss_mcp.graph_rag_agent.GraphStore", return_value=mock_graph_store_instance
    )

    # Mock Vector Store and Retriever
    mock_vector_store = MagicMock()
    mock_vector_retriever = MagicMock()
    # The retriever should return a list of Document objects
    mock_vector_retriever.invoke.return_value = [
        MagicMock(page_content="Vector context")
    ]
    mock_vector_store.as_retriever.return_value = mock_vector_retriever
    mocker.patch("rss_mcp.graph_rag_agent.Neo4jVector", return_value=mock_vector_store)

    # Mock Graph Cypher Chain
    mock_cypher_chain = MagicMock()
    mock_cypher_chain.invoke.return_value = {"result": "Graph context"}
    mocker.patch(
        "rss_mcp.graph_rag_agent.GraphCypherQAChain.from_llm",
        return_value=mock_cypher_chain,
    )

    return {
        "vector_retriever": mock_vector_retriever,
        "cypher_chain": mock_cypher_chain,
        "llm": mock_llm,
    }


def test_hybrid_agent_retrieval_steps(mock_dependencies):
    """
    Tests that the agent's query method invokes both the vector retriever
    and the cypher chain.
    """
    agent = GraphRAGAgent()
    question = "Who is Jules and what do they do?"

    # Act
    result = agent.query(question)

    # Assert
    # Check that both retrieval mechanisms were called with the question
    mock_dependencies["vector_retriever"].invoke.assert_called_once_with(question)
    mock_dependencies["cypher_chain"].invoke.assert_called_once_with(
        {"query": question}
    )

    # Check that the final LLM call was made
    # The mock_llm itself is called by the chain
    mock_dependencies["llm"].assert_called_once()

    # Check that the final result is what the mocked LLM's content was, after parsing
    assert result == "A final combined answer."
