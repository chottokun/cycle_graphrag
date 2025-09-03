import pytest
from unittest.mock import MagicMock, patch

from rss_mcp.embedding_manager import EmbeddingManager

@pytest.fixture(autouse=True)
def reset_singleton():
    """Fixture to reset the singleton instance before each test."""
    EmbeddingManager._reset_for_testing()

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks the dependencies for EmbeddingManager."""
    # Mock ConfigManager
    mock_config_manager_instance = MagicMock()
    mock_config_manager_instance.get_embedding_model_name.return_value = "test-model"
    mocker.patch("rss_mcp.embedding_manager.ConfigManager", return_value=mock_config_manager_instance)

    # Mock HuggingFaceEmbeddings
    mock_hf_embeddings_instance = MagicMock()
    mock_hf_embeddings_class = mocker.patch(
        "rss_mcp.embedding_manager.HuggingFaceEmbeddings",
        return_value=mock_hf_embeddings_instance
    )

    return {
        "config_manager_instance": mock_config_manager_instance,
        "hf_embeddings_class": mock_hf_embeddings_class,
        "hf_embeddings_instance": mock_hf_embeddings_instance
    }

def test_embedding_manager_is_singleton(mock_dependencies):
    """Tests that EmbeddingManager is a singleton."""
    manager1 = EmbeddingManager()
    manager2 = EmbeddingManager()
    assert manager1 is manager2

def test_get_model_initializes_and_caches_model(mock_dependencies):
    """Tests that get_model initializes and returns the correct model."""
    manager = EmbeddingManager()

    # Call get_model for the first time
    model1 = manager.get_model()

    # Assert that the underlying HuggingFaceEmbeddings class was called correctly
    mock_dependencies["hf_embeddings_class"].assert_called_once_with(
        model_name="test-model"
    )
    assert model1 == mock_dependencies["hf_embeddings_instance"]

    # Call get_model a second time
    model2 = manager.get_model()

    # Assert that the constructor was NOT called again (cached result)
    mock_dependencies["hf_embeddings_class"].assert_called_once()
    assert model2 is model1
