import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

# This import will fail initially
from rss_mcp.document_processor import DocumentProcessor


@pytest.fixture
def mock_loaders(mocker):
    """Mocks the document loader and text splitter."""
    # Mock Document object to be returned by the loader
    mock_doc = Document(page_content="Full text of the markdown file.")

    # Mock the loader instance and its load method
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [mock_doc]
    mock_loader_class = mocker.patch(
        "rss_mcp.document_processor.UnstructuredMarkdownLoader",
        return_value=mock_loader_instance,
    )

    # Mock the splitter instance and its split_documents method
    mock_splitter_instance = MagicMock()
    mock_splitter_instance.split_documents.return_value = [
        Document(page_content="Chunk 1"),
        Document(page_content="Chunk 2"),
    ]
    mock_splitter_class = mocker.patch(
        "rss_mcp.document_processor.RecursiveCharacterTextSplitter",
        return_value=mock_splitter_instance,
    )

    return {
        "loader_class": mock_loader_class,
        "loader_instance": mock_loader_instance,
        "splitter_class": mock_splitter_class,
        "splitter_instance": mock_splitter_instance,
    }


def test_process_file_loads_and_splits_document(mock_loaders):
    """
    Tests that the document processor correctly initializes and calls
    the loader and splitter.
    """
    processor = DocumentProcessor()
    file_path = "dummy/path/to/file.md"

    result_chunks = processor.process_file(file_path)

    # Verify loader was called with the correct file path
    mock_loaders["loader_class"].assert_called_once_with(file_path)
    mock_loaders["loader_instance"].load.assert_called_once()

    # Verify splitter was initialized (can add specific args later)
    mock_loaders["splitter_class"].assert_called_once_with(
        chunk_size=1000, chunk_overlap=200
    )

    # Get the document that the loader "returned"
    loaded_docs = mock_loaders["loader_instance"].load.return_value
    # Verify splitter was called with the documents from the loader
    mock_loaders["splitter_instance"].split_documents.assert_called_once_with(
        loaded_docs
    )

    # Verify the final result is what the splitter "returned"
    assert len(result_chunks) == 2
    assert result_chunks[0].page_content == "Chunk 1"
    assert result_chunks[1].page_content == "Chunk 2"
    assert (
        result_chunks == mock_loaders["splitter_instance"].split_documents.return_value
    )
