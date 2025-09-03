from langchain.chains import GraphCypherQAChain
from langchain_community.vectorstores import Neo4jVector
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
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

        llm = llm_manager.get_llm()

        vector_store = Neo4jVector(
            embedding=embedding_manager.get_model(),
            url=graph_store.neo4j_config.uri,
            username=graph_store.neo4j_config.username,
            password=graph_store.neo4j_config.password,
            index_name="chunk_embeddings",
            node_label="Chunk",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
        vector_retriever = vector_store.as_retriever()

        cypher_chain = GraphCypherQAChain.from_llm(
            graph=graph_store.graph,
            llm=llm,
            verbose=True,
            return_intermediate_steps=True,
        )

        final_prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant. Based on the following context from a knowledge graph and a vector search, answer the user's question.
            Vector Search Context: {vector_context}
            Graph Search Context: {graph_context}
            Question: {question}
            Answer:
            """
        )

        # 1. A chain that retrieves parallel contexts
        retrieval_chain = RunnableParallel(
            {
                "vector_context": lambda x: vector_retriever.invoke(x["question"]),
                "graph_context_full": lambda x: cypher_chain.invoke(
                    {"query": x["question"]}
                ),
                "question": lambda x: x["question"],
            }
        )

        # 2. The final chain that uses the retrieved context to generate an answer
        answer_generation_chain = (
            RunnablePassthrough.assign(
                graph_context=lambda x: x["graph_context_full"]["result"]
            )
            | final_prompt
            | llm
            | StrOutputParser()
        )

        # 3. The final parallel chain to produce the desired output format
        self.chain = retrieval_chain | RunnableParallel(
            {
                "answer": answer_generation_chain,
                "context": lambda x: x["graph_context_full"]["intermediate_steps"][0][
                    "context"
                ],
            }
        )

    def query(self, question: str) -> dict:
        """
        Asks a question using the hybrid RAG chain and returns the answer and context.

        Args:
            question: The question to ask.

        Returns:
            A dictionary containing the 'answer' and the 'context'.
        """
        print(f"Querying with hybrid RAG approach: '{question}'")
        return self.chain.invoke({"question": question})
