from langchain.chains import GraphCypherQAChain
from langchain_community.vectorstores import Neo4jVector
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from .llm_manager import LLMManager
from .graph_store import GraphStore
from .embedding_manager import EmbeddingManager


class GraphRAGAgent:
    """
    An agent that uses a hybrid RAG approach (vector + graph) to answer questions.
    """

    def __init__(self):
        """
        Initializes the GraphRAGAgent and its components.
        """
        llm_manager = LLMManager()
        graph_store = GraphStore()
        embedding_manager = EmbeddingManager()

        self.graph = graph_store.graph
        self.llm = llm_manager.get_llm()
        self.embedding_model = embedding_manager.get_model()

        # Initialize the vector store for semantic search
        self.vector_store = Neo4jVector(
            embedding=self.embedding_model,
            url=graph_store.neo4j_config.uri,
            username=graph_store.neo4j_config.username,
            password=graph_store.neo4j_config.password,
            index_name="chunk_embeddings",
            node_label="Chunk",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
        self.vector_retriever = self.vector_store.as_retriever()

        # Initialize the graph-based QA chain for structured queries
        self.cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph, llm=self.llm, verbose=True
        )

        # Define the final prompt template to combine contexts
        self.final_prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant. Based on the following context from a knowledge graph and a vector search, answer the user's question.
            The vector search provides general context, while the graph search provides specific, structured information.
            Use the graph context primarily for specific facts and relationships, and the vector context for broader understanding.

            Vector Search Context:
            {vector_context}

            Graph Search Context:
            {graph_context}

            Question:
            {question}

            Answer:
            """
        )

        # Build the final combined chain using LCEL
        self.chain = (
            RunnablePassthrough.assign(
                vector_context=lambda x: self.vector_retriever.invoke(x["question"]),
                graph_context=lambda x: self.cypher_chain.invoke(
                    {"query": x["question"]}
                )["result"],
            )
            | self.final_prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        """
        Asks a question using the hybrid RAG chain and returns the answer.

        Args:
            question: The question to ask.

        Returns:
            The answer from the RAG agent.
        """
        print(f"Querying with hybrid RAG approach: '{question}'")
        return self.chain.invoke({"question": question})
