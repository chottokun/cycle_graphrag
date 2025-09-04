import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument

# This import will fail initially
from rss_mcp.graph_converter import GraphConverter


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks the LLMManager and LLMGraphTransformer."""
    # Mock LLM and LLMManager
    mock_llm = MagicMock()
    mock_llm_manager_instance = MagicMock()
    mock_llm_manager_instance.get_llm.return_value = mock_llm
    mocker.patch(
        "rss_mcp.graph_converter.LLMManager", return_value=mock_llm_manager_instance
    )

    # Mock GraphDocument objects to be returned by the transformer
    mock_graph_doc = GraphDocument(
        nodes=[], relationships=[], source=Document(page_content="")
    )

    # Mock the transformer instance and its conversion method
    mock_transformer_instance = MagicMock()
    mock_transformer_instance.convert_to_graph_documents.return_value = [mock_graph_doc]
    mock_transformer_class = mocker.patch(
        "rss_mcp.graph_converter.LLMGraphTransformer",
        return_value=mock_transformer_instance,
    )

    return {
        "llm_manager_instance": mock_llm_manager_instance,
        "llm": mock_llm,
        "transformer_class": mock_transformer_class,
        "transformer_instance": mock_transformer_instance,
        "graph_doc": mock_graph_doc,
    }


def test_convert_to_graph(mock_dependencies):
    """
    Tests that the graph converter correctly uses its dependencies
    to convert document chunks to graph documents.
    """
    # 1. Arrange
    # Create a converter instance. Its __init__ will use the mocked LLMManager.
    converter = GraphConverter(llm_name="test_llm")

    # Create some dummy document chunks
    doc_chunks = [
        Document(page_content="Chunk 1 about Jules."),
        Document(page_content="Chunk 2 about software engineering."),
    ]

    # 2. Act
    result_graph_docs = converter.convert_to_graph(doc_chunks)

    # 3. Assert
    # Verify LLMManager was used to get the correct LLM
    mock_dependencies["llm_manager_instance"].get_llm.assert_called_once_with(
        "test_llm"
    )

    # Verify the transformer was initialized with the LLM from the manager
    mock_dependencies["transformer_class"].assert_called_once_with(
        llm=mock_dependencies["llm"]
    )

    # Verify the conversion method was called with the right chunks
    mock_dependencies[
        "transformer_instance"
    ].convert_to_graph_documents.assert_called_once_with(doc_chunks)

    # Verify the result is what the transformer returned
    assert result_graph_docs == [mock_dependencies["graph_doc"]]
