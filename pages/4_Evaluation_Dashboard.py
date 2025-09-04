import streamlit as st
from dotenv import load_dotenv
import os

from rss_mcp.evaluation import run_evaluation

# --- Streamlit Page ---
st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊")
st.title("📊 RAG Evaluation Dashboard")

st.markdown(
    """
    This page allows you to evaluate the performance of the RAG system using the `ragas` library.
    The evaluation is run against a predefined dataset (`data/eval_dataset.jsonl`).

    **Metrics:**
    - **faithfulness**: Measures how factually consistent the generated answer is with the retrieved context.
    - **answer_relevancy**: Measures how relevant the answer is to the question.
    - **context_recall**: Measures the ability of the retriever to retrieve all necessary information.
    - **context_precision**: Measures how relevant the retrieved context is to the question.
    """
)

# Load environment variables
load_dotenv()

EVAL_DATASET_PATH = "data/eval_dataset.jsonl"

if not os.path.exists(EVAL_DATASET_PATH):
    st.error(f"Evaluation dataset not found at `{EVAL_DATASET_PATH}`.")
    st.stop()

if st.button("🚀 Run Evaluation", type="primary"):
    st.info(
        "Starting evaluation... This may take several minutes depending on the dataset size and the LLM."
    )

    try:
        with st.spinner("Running evaluation in progress... Please wait."):
            results_df = run_evaluation(EVAL_DATASET_PATH)

        st.success("✅ Evaluation Complete!")

        st.subheader("📊 Overall Scores")

        # Calculate mean scores for key metrics
        mean_scores = results_df[
            ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        ].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faithfulness", f"{mean_scores['faithfulness']:.2f}")
        col2.metric("Answer Relevancy", f"{mean_scores['answer_relevancy']:.2f}")
        col3.metric("Context Precision", f"{mean_scores['context_precision']:.2f}")
        col4.metric("Context Recall", f"{mean_scores['context_recall']:.2f}")

        st.subheader("📈 Detailed Results per Question")
        st.dataframe(results_df)

    except Exception as e:
        st.error(f"An error occurred during evaluation: {e}")
        st.warning(
            "Please ensure all services (Neo4j, Ollama) are running and that an API key for OpenAI is set if required by Ragas."
        )
else:
    st.info("Click the button above to start the evaluation process.")
