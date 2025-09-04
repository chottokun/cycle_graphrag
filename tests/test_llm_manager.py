import pytest

from rss_mcp.llm_manager import LLMManager
from rss_mcp.config_manager import ConfigManager


@pytest.fixture(autouse=True)
def reset_singletons():
    """Resets all singletons before each test."""
    ConfigManager._reset_for_testing()
    LLMManager._reset_for_testing()


def test_llm_manager_is_singleton():
    """Tests that LLMManager follows the singleton pattern."""
    manager1 = LLMManager()
    manager2 = LLMManager()
    assert manager1 is manager2


def test_get_default_llm_returns_ollama(mocker):
    """Tests that the default LLM is loaded correctly (Ollama)."""
    # Patch ChatOllama in the module where it's used (llm_manager)
    mock_chat_ollama = mocker.patch("rss_mcp.llm_manager.ChatOllama")

    manager = LLMManager()
    llm_instance = manager.get_llm()

    # Check that ChatOllama was called once with the correct parameters
    mock_chat_ollama.assert_called_once_with(
        model="llama3", base_url="http://localhost:11434"
    )
    # Check that the returned instance is the one created by the mock
    assert llm_instance == mock_chat_ollama.return_value


def test_get_named_llm_returns_ollama(mocker):
    """Tests getting a specifically named Ollama LLM."""
    mock_chat_ollama = mocker.patch("rss_mcp.llm_manager.ChatOllama")

    manager = LLMManager()
    manager.get_llm("ollama_llama3")

    mock_chat_ollama.assert_called_once_with(
        model="llama3", base_url="http://localhost:11434"
    )


def test_llm_instances_are_cached(mocker):
    """Tests that LLM instances are cached and not recreated."""
    mock_chat_ollama = mocker.patch("rss_mcp.llm_manager.ChatOllama")

    manager = LLMManager()
    instance1 = manager.get_llm("ollama_llama3")
    instance2 = manager.get_llm("ollama_llama3")

    # The constructor should only be called once
    mock_chat_ollama.assert_called_once()
    assert instance1 is instance2


def test_get_llm_returns_azure(mocker):
    """Tests that an Azure OpenAI LLM is loaded correctly."""
    # Mock the environment variables needed by AzureChatOpenAI
    mocker.patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "OPENAI_API_VERSION": "2023-05-15",
        },
    )
    # Patch the AzureChatOpenAI class to monitor its instantiation
    mock_azure_chat = mocker.patch("rss_mcp.llm_manager.AzureChatOpenAI")

    manager = LLMManager()
    llm_instance = manager.get_llm("azure_gpt4")

    # Check that AzureChatOpenAI was called once with the correct parameters
    mock_azure_chat.assert_called_once_with(
        azure_deployment="gpt-4o",  # from config.toml model_name
        api_version="2023-05-15",
    )
    assert llm_instance == mock_azure_chat.return_value


def test_get_non_existent_llm_raises_error():
    """Tests that a KeyError is raised for a non-existent LLM."""
    manager = LLMManager()
    with pytest.raises(KeyError):
        manager.get_llm("this_llm_does_not_exist")
