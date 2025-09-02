from langchain.chains import GraphCypherQAChain

from .llm_manager import LLMManager
from .graph_store import GraphStore


class GraphRAGAgent:
    """
    An agent that uses a knowledge graph to answer questions.
    """

    def __init__(self):
        """
        Initializes the GraphRAGAgent.
        """
        llm_manager = LLMManager()
        graph_store = GraphStore()

        # Get the underlying Neo4jGraph instance from the GraphStore
        graph = graph_store.graph

        # Get the default LLM for the agent
        llm = llm_manager.get_llm()

        # Create the QA chain
        self.chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

    def query(self, question: str) -> str:
        """
        Asks a question to the knowledge graph and returns the answer.

        Args:
            question: The question to ask.

        Returns:
            The answer from the RAG agent.
        """
        print(f"Querying the graph with: '{question}'")
        result = self.chain.invoke({"query": question})
        # The result is a dictionary, e.g., {'result': 'The answer.'}
        return result.get("result", "No answer found.")
