import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from rss_mcp.graph_rag_agent import GraphRAGAgent


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks the core dependencies for the Hybrid GraphRAGAgent."""
    # Mock Managers
    mock_llm_output = AIMessage(content="A final combined answer.")
    mock_llm = MagicMock(side_effect=lambda *args, **kwargs: mock_llm_output)
    mock_llm_manager_instance = MagicMock(get_llm=lambda: mock_llm)
    mocker.patch(
        "rss_mcp.graph_rag_agent.LLMManager", return_value=mock_llm_manager_instance
    )

    mock_embedding_model = MagicMock()
    mock_embedding_manager_instance = MagicMock(get_model=lambda: mock_embedding_model)
    mocker.patch(
        "rss_mcp.graph_rag_agent.EmbeddingManager",
        return_value=mock_embedding_manager_instance,
    )

    # Mock Neo4jGraph and GraphStore
    mock_neo4j_graph = MagicMock()
    mock_graph_store_instance = MagicMock(graph=mock_neo4j_graph)
    mocker.patch(
        "rss_mcp.graph_rag_agent.GraphStore", return_value=mock_graph_store_instance
    )

    # Mock Vector Store and Retriever
    mock_vector_retriever = MagicMock()
    mock_vector_retriever.invoke.return_value = [
        MagicMock(page_content="Vector context")
    ]
    mock_vector_store_instance = MagicMock(as_retriever=lambda: mock_vector_retriever)
    mocker.patch(
        "rss_mcp.graph_rag_agent.Neo4jVector", return_value=mock_vector_store_instance
    )

    # Mock Graph Cypher Chain to return intermediate steps
    mock_cypher_chain = MagicMock()
    mock_cypher_chain.invoke.return_value = {
        "result": "Graph context",
        "intermediate_steps": [{"context": "some graph data"}],
    }
    mocker.patch(
        "rss_mcp.graph_rag_agent.GraphCypherQAChain.from_llm",
        return_value=mock_cypher_chain,
    )

    return {
        "vector_retriever": mock_vector_retriever,
        "cypher_chain": mock_cypher_chain,
        "llm": mock_llm,
    }


def test_hybrid_agent_returns_answer_and_context(mock_dependencies):
    """
    Tests that the agent's query method returns a dictionary with
    both the final answer and the intermediate graph context.
    """
    agent = GraphRAGAgent()
    question = "Who is Jules?"

    # Act
    result = agent.query(question)

    # Assert
    assert isinstance(result, dict)
    assert "answer" in result
    assert "context" in result
    assert result["answer"] == "A final combined answer."
    assert result["context"] == "some graph data"

    # Verify dependencies were called
    mock_dependencies["vector_retriever"].invoke.assert_called_once_with(question)
    mock_dependencies["cypher_chain"].invoke.assert_called_once_with(
        {"query": question}
    )
    # The llm mock itself is called by the chain
    mock_dependencies["llm"].assert_called_once()
