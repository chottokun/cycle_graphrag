import pytest
from unittest.mock import MagicMock
from datasets import Dataset
import pandas as pd

# This import will fail initially
from rss_mcp.evaluation import run_evaluation


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks the dependencies for the evaluation logic."""
    # Mock the RAG agent
    mock_agent_instance = MagicMock()
    # The agent needs to return a dict with answer and context
    mock_agent_instance.query.return_value = {
        "answer": "mocked answer",
        "context": ["mocked context"],
    }
    mocker.patch("rss_mcp.evaluation.GraphRAGAgent", return_value=mock_agent_instance)

    # Mock the ragas.evaluate function to return an object with a .to_pandas() method
    mock_pandas_df = pd.DataFrame({"faithfulness": [1.0], "answer_relevancy": [1.0]})
    mock_ragas_result = MagicMock()
    mock_ragas_result.to_pandas.return_value = mock_pandas_df
    mock_ragas_evaluate = mocker.patch(
        "rss_mcp.evaluation.evaluate", return_value=mock_ragas_result
    )

    return {
        "agent_instance": mock_agent_instance,
        "ragas_evaluate": mock_ragas_evaluate,
        "evaluate_result": mock_pandas_df,
    }


def test_run_evaluation(mock_dependencies):
    """
    Tests that the run_evaluation function correctly prepares data and
    calls the ragas.evaluate function.
    """
    dataset_path = "data/eval_dataset.jsonl"

    # Act
    result_df = run_evaluation(dataset_path)

    # Assert
    # Check that the agent was called for each question in the dataset (3 times)
    assert mock_dependencies["agent_instance"].query.call_count == 3

    # Check that ragas.evaluate was called once
    mock_dependencies["ragas_evaluate"].assert_called_once()

    # Inspect the arguments passed to ragas.evaluate
    args, kwargs = mock_dependencies["ragas_evaluate"].call_args
    assert "dataset" in kwargs
    eval_dataset = kwargs["dataset"]
    assert isinstance(eval_dataset, Dataset)
    assert len(eval_dataset) == 3
    # Check that the dataset was augmented with 'answer' and 'contexts'
    assert "answer" in eval_dataset.features
    assert "contexts" in eval_dataset.features

    # Check that the final result is what the mocked function returned
    pd.testing.assert_frame_equal(result_df, mock_dependencies["evaluate_result"])
