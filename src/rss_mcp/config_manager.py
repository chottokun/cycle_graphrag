import tomli
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    """Dataclass to hold LLM configuration."""

    provider: str
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    # Add other potential fields with default None
    api_key: Optional[str] = None


class ConfigManager:
    """
    Manages loading and accessing configuration from a TOML file.
    Follows the Singleton pattern to ensure a single instance.
    """

    _instance: Optional["ConfigManager"] = None
    _config: Dict[str, Any] = {}
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "config.toml"):
        if self._initialized:
            return
        self._load_config(config_path)
        self._initialized = True

    def _load_config(self, config_path: str):
        """Loads the configuration from the specified TOML file."""
        try:
            with open(config_path, "rb") as f:
                self._config = tomli.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        except tomli.TOMLDecodeError:
            raise ValueError(f"Could not decode the TOML file at: {config_path}")

    def get_default_llm_name(self) -> str:
        """Returns the name of the default LLM specified in the config."""
        default_name = self._config.get("default_llm")
        if not default_name:
            raise ValueError("'default_llm' not specified in the configuration file.")
        return default_name

    def get_llm_config(self, name: str) -> LLMConfig:
        """
        Returns the configuration for a specific LLM by its name.
        """
        try:
            # Access the nested dictionary e.g., [llm.ollama_llama3]
            config_dict = self._config["llm"][name]
        except KeyError:
            raise KeyError(
                f"LLM configuration for '{name}' not found in the config file."
            )

        if "provider" not in config_dict:
            raise ValueError(
                f"LLM configuration for '{name}' must include a 'provider' key."
            )

        return LLMConfig(
            provider=config_dict.get("provider"),
            model_name=config_dict.get("model_name"),
            base_url=config_dict.get("base_url"),
            api_key=config_dict.get("api_key"),
        )

    @classmethod
    def _reset_for_testing(cls):
        """Resets the singleton instance and its initialized state. For testing only."""
        cls._instance = None
        cls._initialized = False
