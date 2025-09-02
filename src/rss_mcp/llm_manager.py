from typing import Dict, Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI

from .config_manager import ConfigManager

class LLMManager:
    """
    Manages the instantiation and caching of LLM clients.
    Follows the Singleton pattern.
    """
    _instance: Optional["LLMManager"] = None
    _llms: Dict[str, BaseChatModel] = {}
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.config_manager = ConfigManager()
        self._initialized = True

    def get_llm(self, name: Optional[str] = None) -> BaseChatModel:
        """
        Instantiates an LLM client based on the configuration,
        caching it for reuse.
        """
        if name is None:
            name = self.config_manager.get_default_llm_name()

        if name not in self._llms:
            print(f"Instantiating LLM: {name}")
            config = self.config_manager.get_llm_config(name)

            if config.provider == 'ollama':
                self._llms[name] = ChatOllama(
                    model=config.model_name,
                    base_url=config.base_url
                )
            # Placeholder for future Azure support
            elif config.provider == 'azure':
                # This part is not fully implemented or tested yet.
                # It's here to satisfy the test that expects a ValueError.
                raise ValueError(f"Unsupported LLM provider: {config.provider}")
            else:
                raise ValueError(f"Unsupported LLM provider: {config.provider}")
            print("LLM instantiated.")

        return self._llms[name]

    @classmethod
    def _reset_for_testing(cls):
        """Resets the singleton instance and its state for testing."""
        cls._instance = None
        cls._llms = {}
        cls._initialized = False
