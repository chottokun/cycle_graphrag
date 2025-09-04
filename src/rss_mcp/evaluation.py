from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import pandas as pd

from .graph_rag_agent import GraphRAGAgent


def run_evaluation(dataset_path: str) -> pd.DataFrame:
    """
    Runs the RAGAS evaluation on the given dataset.

    Args:
        dataset_path: The path to the .jsonl evaluation dataset.

    Returns:
        A pandas DataFrame with the evaluation results.
    """
    print("--- Running RAGAS Evaluation ---")

    # 1. Load the ground truth dataset
    print(f"Loading dataset from: {dataset_path}")
    eval_dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 2. Generate answers and contexts using the RAG agent
    print("Initializing RAG agent to generate answers...")
    agent = GraphRAGAgent()

    generated_data = []
    for row in eval_dataset:
        question = row["question"]
        print(f"  - Querying agent for question: '{question}'")
        response = agent.query(question)

        # We need to format the context for Ragas. It expects a list of strings.
        # The agent returns a list of dicts. We'll extract the 'text' property.
        contexts = []
        if response["context"]:
            # The context can be complex, let's try to find text properties
            for item in response["context"]:
                if (
                    isinstance(item, dict)
                    and "properties" in item
                    and "text" in item["properties"]
                ):
                    contexts.append(item["properties"]["text"])

        generated_data.append({"answer": response["answer"], "contexts": contexts})

    # Add the generated 'answer' and 'contexts' to the dataset
    generated_df = pd.DataFrame(generated_data)
    # The original dataset has a 'contexts' column which is the ground truth context.
    # Ragas expects the 'contexts' column to be the retrieved context.
    # So, we remove the old one before adding the new one.
    eval_dataset = eval_dataset.remove_columns("contexts")
    eval_dataset = eval_dataset.add_column("answer", generated_df["answer"])
    eval_dataset = eval_dataset.add_column("contexts", generated_df["contexts"])

    # 3. Run RAGAS evaluation
    print("Running ragas.evaluate...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        raise_exceptions=False,  # To prevent stopping on a single failed evaluation
    )

    print("--- Evaluation Complete ---")
    return result.to_pandas()
