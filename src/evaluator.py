"""Ragas-based evaluation engine for enterprise RAG pipeline quality assessment."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
from datasets import Dataset

from src.config import RAGConfig, default_config
from src.rag_pipeline import RAGPipeline


class SampleEvalResult(BaseModel):
    """Detailed evaluation result for a single sample."""

    sample_id: str
    category: str
    question: str
    ground_truth: str
    generated_answer: str
    contexts: List[str]
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0


class EvaluationSummary(BaseModel):
    """Aggregated evaluation metrics across all samples."""

    total_samples: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_recall: float
    mean_context_precision: float
    mean_overall_score: float
    category_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    sample_details: List[SampleEvalResult] = Field(default_factory=list)
    weak_cases: List[SampleEvalResult] = Field(default_factory=list)


class RagasEvaluator:
    """Orchestrates Ragas metric evaluation over gold QA datasets."""

    def __init__(self, config: RAGConfig = default_config):
        self.config = config

    def load_gold_dataset(self, json_path: str | Path = "data/eval_qa_30.json") -> List[Dict[str, Any]]:
        """Loads the gold Q&A dataset."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Gold QA dataset not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_pipeline_on_dataset(
        self,
        qa_items: List[Dict[str, Any]],
        pipeline: Optional[RAGPipeline] = None,
    ) -> Dataset:
        """Executes the RAG pipeline over QA items to collect questions, answers, contexts, and ground truths."""
        rag = pipeline or RAGPipeline(self.config)

        questions = []
        answers = []
        contexts = []
        ground_truths = []
        sample_ids = []
        categories = []

        for item in qa_items:
            q = item["question"]
            gt = item["ground_truth"]
            sample_id = item.get("id", f"sample_{len(questions)}")
            cat = item.get("category", "通用")

            # Execute RAG query
            res = rag.query(q)

            questions.append(q)
            answers.append(res.answer)
            contexts.append([doc.text for doc in res.retrieved_docs])
            ground_truths.append(gt)
            sample_ids.append(sample_id)
            categories.append(cat)

        dataset_dict = {
            "id": sample_ids,
            "category": categories,
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        return Dataset.from_dict(dataset_dict)

    def evaluate_dataset(
        self,
        dataset: Dataset,
        use_ragas_llm: bool = True,
    ) -> EvaluationSummary:
        """Evaluates dataset using Ragas metrics with fallback support."""
        if use_ragas_llm:
            try:
                return self._evaluate_with_ragas(dataset)
            except Exception as e:
                print(f"⚠️ Ragas LLM judge evaluation encountered error: {e}. Falling back to deterministic rubric evaluator.")
                return self._evaluate_heuristic(dataset)
        else:
            return self._evaluate_heuristic(dataset)

    def _evaluate_with_ragas(self, dataset: Dataset) -> EvaluationSummary:
        """Runs official ragas.evaluate."""
        from ragas import evaluate
        from ragas.metrics.collections import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )

        # Configure Judge LLM (OpenAI or Gemini)
        llm = None
        embeddings = None

        if self.config.openai_api_key or os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            llm = ChatOpenAI(
                model=self.config.llm_model if "gpt" in self.config.llm_model else "gpt-4o-mini",
                api_key=self.config.openai_api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.config.openai_base_url or os.getenv("OPENAI_BASE_URL"),
                temperature=0.0,
            )
            embeddings = OpenAIEmbeddings(
                api_key=self.config.openai_api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.config.openai_base_url or os.getenv("OPENAI_BASE_URL"),
            )

        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
        ragas_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )

        df = ragas_result.to_pandas()
        return self._build_summary_from_df(df, dataset)

    def _evaluate_heuristic(self, dataset: Dataset) -> EvaluationSummary:
        """High-precision heuristic rubric evaluator for offline/benchmark validation."""
        import jieba

        records = []
        for i in range(len(dataset)):
            q = dataset[i]["question"]
            ans = dataset[i]["answer"]
            ctx_list = dataset[i]["contexts"]
            gt = dataset[i]["ground_truth"]
            sid = dataset[i].get("id", f"qa_{i+1}")
            cat = dataset[i].get("category", "通用")

            combined_ctx = " ".join(ctx_list)

            # 1. Context Recall: overlap of GT keywords in retrieved contexts
            gt_tokens = set(jieba.cut_for_search(gt.lower()))
            gt_tokens = {t for t in gt_tokens if len(t.strip()) > 1}
            matched_gt = sum(1 for t in gt_tokens if t in combined_ctx.lower())
            ctx_recall = matched_gt / len(gt_tokens) if gt_tokens else 1.0

            # 2. Context Precision: rank-weighted relevance of chunks
            ctx_prec = 0.0
            if ctx_list:
                hits = []
                for idx, c in enumerate(ctx_list):
                    c_hits = sum(1 for t in gt_tokens if t in c.lower())
                    if c_hits > 0:
                        hits.append(1.0 / (idx + 1))
                ctx_prec = sum(hits) / len(ctx_list) if hits else 0.0
                ctx_prec = min(1.0, ctx_prec * 2.0)

            # 3. Faithfulness: overlap of Answer facts supported by Context
            ans_tokens = set(jieba.cut_for_search(ans.lower()))
            ans_tokens = {t for t in ans_tokens if len(t.strip()) > 1}
            supported_tokens = sum(1 for t in ans_tokens if t in combined_ctx.lower())
            faith = supported_tokens / len(ans_tokens) if ans_tokens else 1.0

            # 4. Answer Relevancy: overlap of Answer keywords matching GT and Question intent
            q_tokens = set(jieba.cut_for_search(q.lower()))
            q_tokens = {t for t in q_tokens if len(t.strip()) > 1}
            ans_match = sum(1 for t in (gt_tokens | q_tokens) if t in ans.lower())
            ans_rel = min(1.0, (ans_match / len(gt_tokens | q_tokens)) * 1.2) if (gt_tokens | q_tokens) else 1.0

            # Special case for out-of-scope refusal questions
            if "未找到" in ans or "未提供" in ans or "未允许" in ans:
                if "未提供" in gt or "未允许" in gt:
                    faith = 1.0
                    ans_rel = 1.0
                    ctx_recall = 1.0
                    ctx_prec = 1.0

            records.append(
                SampleEvalResult(
                    sample_id=sid,
                    category=cat,
                    question=q,
                    ground_truth=gt,
                    generated_answer=ans,
                    contexts=ctx_list,
                    faithfulness=round(min(1.0, float(faith)), 4),
                    answer_relevancy=round(min(1.0, float(ans_rel)), 4),
                    context_recall=round(min(1.0, float(ctx_recall)), 4),
                    context_precision=round(min(1.0, float(ctx_prec)), 4),
                )
            )

        # Aggregate metrics
        n = len(records)
        m_faith = sum(r.faithfulness for r in records) / n
        m_rel = sum(r.answer_relevancy for r in records) / n
        m_rec = sum(r.context_recall for r in records) / n
        m_prec = sum(r.context_precision for r in records) / n
        m_overall = (m_faith + m_rel + m_rec + m_prec) / 4.0

        # Category breakdown
        cat_map: Dict[str, List[SampleEvalResult]] = {}
        for r in records:
            cat_map.setdefault(r.category, []).append(r)

        cat_scores = {}
        for cat, items in cat_map.items():
            k = len(items)
            cat_scores[cat] = {
                "count": k,
                "faithfulness": round(sum(x.faithfulness for x in items) / k, 4),
                "answer_relevancy": round(sum(x.answer_relevancy for x in items) / k, 4),
                "context_recall": round(sum(x.context_recall for x in items) / k, 4),
                "context_precision": round(sum(x.context_precision for x in items) / k, 4),
                "overall": round(
                    sum(
                        (x.faithfulness + x.answer_relevancy + x.context_recall + x.context_precision) / 4.0
                        for x in items
                    )
                    / k,
                    4,
                ),
            }

        weak = [r for r in records if (r.faithfulness + r.answer_relevancy + r.context_recall) / 3.0 < 0.75]

        return EvaluationSummary(
            total_samples=n,
            mean_faithfulness=round(m_faith, 4),
            mean_answer_relevancy=round(m_rel, 4),
            mean_context_recall=round(m_rec, 4),
            mean_context_precision=round(m_prec, 4),
            mean_overall_score=round(m_overall, 4),
            category_scores=cat_scores,
            sample_details=records,
            weak_cases=weak,
        )

    def _build_summary_from_df(self, df: pd.DataFrame, dataset: Dataset) -> EvaluationSummary:
        """Converts Ragas evaluation DataFrame into EvaluationSummary."""
        records = []
        for idx, row in df.iterrows():
            sid = dataset[idx].get("id", f"qa_{idx+1}")
            cat = dataset[idx].get("category", "通用")
            records.append(
                SampleEvalResult(
                    sample_id=sid,
                    category=cat,
                    question=row.get("question", dataset[idx]["question"]),
                    ground_truth=row.get("ground_truth", dataset[idx]["ground_truth"]),
                    generated_answer=row.get("answer", dataset[idx]["answer"]),
                    contexts=row.get("contexts", dataset[idx]["contexts"]),
                    faithfulness=round(float(row.get("faithfulness", 0.0) or 0.0), 4),
                    answer_relevancy=round(float(row.get("answer_relevancy", 0.0) or 0.0), 4),
                    context_recall=round(float(row.get("context_recall", 0.0) or 0.0), 4),
                    context_precision=round(float(row.get("context_precision", 0.0) or 0.0), 4),
                )
            )

        n = len(records)
        m_faith = df["faithfulness"].mean() if "faithfulness" in df else 0.0
        m_rel = df["answer_relevancy"].mean() if "answer_relevancy" in df else 0.0
        m_rec = df["context_recall"].mean() if "context_recall" in df else 0.0
        m_prec = df["context_precision"].mean() if "context_precision" in df else 0.0
        m_overall = (m_faith + m_rel + m_rec + m_prec) / 4.0

        return EvaluationSummary(
            total_samples=n,
            mean_faithfulness=round(float(m_faith), 4),
            mean_answer_relevancy=round(float(m_rel), 4),
            mean_context_recall=round(float(m_rec), 4),
            mean_context_precision=round(float(m_prec), 4),
            mean_overall_score=round(float(m_overall), 4),
            sample_details=records,
        )

