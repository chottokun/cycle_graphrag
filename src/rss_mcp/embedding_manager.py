from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings

from .config_manager import ConfigManager


class EmbeddingManager:
    """
    Manages the instantiation and caching of embedding models.
    Follows the Singleton pattern.
    """

    _instance: Optional["EmbeddingManager"] = None
    _model: Optional[HuggingFaceEmbeddings] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.config_manager = ConfigManager()
        self._initialized = True

    def get_model(self) -> HuggingFaceEmbeddings:
        """
        Initializes and returns the embedding model, caching it for reuse.
        """
        if self._model is None:
            model_name = self.config_manager.get_embedding_model_name()
            print(f"Initializing embedding model: {model_name}")
            self._model = HuggingFaceEmbeddings(model_name=model_name)
            print("Embedding model initialized.")

        return self._model

    @classmethod
    def _reset_for_testing(cls):
        """Resets the singleton instance and its state for testing."""
        cls._instance = None
        cls._model = None
        cls._initialized = False
