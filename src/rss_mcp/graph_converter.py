from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer

from .llm_manager import LLMManager


class GraphConverter:
    """
    Converts document chunks into graph documents using an LLM.
    """

    def __init__(self, llm_name: Optional[str] = None):
        """
        Initializes the converter.

        Args:
            llm_name: The name of the LLM to use from the LLMManager.
                      If None, the default LLM is used.
        """
        llm_manager = LLMManager()
        llm = llm_manager.get_llm(llm_name)
        self.transformer = LLMGraphTransformer(llm=llm)

    def convert_to_graph(self, documents: List[Document]) -> List[GraphDocument]:
        """
        Converts a list of document chunks into a list of graph documents.

        Args:
            documents: A list of Document objects (chunks).

        Returns:
            A list of GraphDocument objects.
        """
        print(f"Converting {len(documents)} chunks to graph documents...")
        graph_documents = self.transformer.convert_to_graph_documents(documents)
        print("Conversion complete.")
        return graph_documents
