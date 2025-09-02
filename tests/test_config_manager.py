import pytest
from unittest.mock import patch, mock_open

# The import now target the classes, not an instance
from rss_mcp.config_manager import ConfigManager, LLMConfig

# Define the path to the config file relative to the project root
CONFIG_FILE_PATH = "config.toml"


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    """Fixture to reset the singleton instance before each test."""
    ConfigManager._reset_for_testing()


def test_config_manager_is_singleton():
    """Tests that ConfigManager follows the singleton pattern."""
    instance1 = ConfigManager(config_path=CONFIG_FILE_PATH)
    instance2 = ConfigManager(config_path=CONFIG_FILE_PATH)
    assert instance1 is instance2, "ConfigManager should be a singleton."


def test_config_file_not_found():
    """Tests that a FileNotFoundError is raised if the config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        ConfigManager(config_path="non_existent_config.toml")


def test_get_default_llm_name():
    """Tests retrieving the default LLM name from the config."""
    manager = ConfigManager(config_path=CONFIG_FILE_PATH)
    assert manager.get_default_llm_name() == "ollama_llama3"


def test_get_llm_config_success():
    """Tests retrieving a specific LLM's configuration."""
    manager = ConfigManager(config_path=CONFIG_FILE_PATH)
    config = manager.get_llm_config("ollama_llama3")
    assert isinstance(config, LLMConfig)
    assert config.provider == "ollama"
    assert config.model_name == "llama3"
    assert config.base_url == "http://localhost:11434"


def test_get_llm_config_not_found():
    """Tests that a KeyError is raised for a non-existent LLM configuration."""
    manager = ConfigManager(config_path=CONFIG_FILE_PATH)
    with pytest.raises(KeyError):
        manager.get_llm_config("non_existent_llm")


def test_get_llm_config_missing_provider():
    """Tests that a ValueError is raised if 'provider' is missing."""
    bad_config_content = """
default_llm = "bad_llm"
[llm.bad_llm]
model_name = "test"
"""
    # Correctly use mock_open from unittest.mock, providing bytes to read_data
    with patch(
        "builtins.open", mock_open(read_data=bad_config_content.encode("utf-8"))
    ):
        # When mocking the file, the manager needs to be re-initialized
        # with a path that will be intercepted by the mock.
        manager = ConfigManager(config_path="any/mocked/path.toml")
        with pytest.raises(ValueError, match="provider"):
            manager.get_llm_config("bad_llm")
