"""Unit and integration tests for Ragas evaluation module."""

import pytest
from pathlib import Path
from src.config import RAGConfig
from src.rag_pipeline import RAGPipeline
from src.evaluator import RagasEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DATASET_PATH = PROJECT_ROOT / "data" / "eval_qa_30.json"


def test_gold_dataset_valid():
    """Verify that the gold QA dataset contains 30 valid items with ground truths."""
    evaluator = RagasEvaluator()
    qa_items = evaluator.load_gold_dataset(GOLD_DATASET_PATH)

    assert len(qa_items) == 30, f"Expected 30 QA items, found {len(qa_items)}"
    for item in qa_items:
        assert "id" in item
        assert "category" in item
        assert "question" in item and len(item["question"]) > 3
        assert "ground_truth" in item and len(item["ground_truth"]) > 5
        assert "target_doc" in item


def test_evaluator_dataset_generation_and_metrics():
    """Verify end-to-end dataset preparation and metric calculations on sample items."""
    config = RAGConfig()
    pipeline = RAGPipeline(config=config)
    evaluator = RagasEvaluator(config=config)

    qa_items = evaluator.load_gold_dataset(GOLD_DATASET_PATH)[:3]
    dataset = evaluator.run_pipeline_on_dataset(qa_items, pipeline=pipeline)

    assert len(dataset) == 3
    assert "question" in dataset.column_names
    assert "answer" in dataset.column_names
    assert "contexts" in dataset.column_names
    assert "ground_truth" in dataset.column_names

    summary = evaluator.evaluate_dataset(dataset, use_ragas_llm=False)

    assert summary.total_samples == 3
    assert 0.0 <= summary.mean_faithfulness <= 1.0
    assert 0.0 <= summary.mean_answer_relevancy <= 1.0
    assert 0.0 <= summary.mean_context_recall <= 1.0
    assert 0.0 <= summary.mean_context_precision <= 1.0
    assert 0.0 <= summary.mean_overall_score <= 1.0
    assert len(summary.sample_details) == 3

