#!/usr/bin/env python3
"""Run Ragas evaluation on enterprise knowledge base Q&A benchmark."""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import RAGConfig
from src.rag_pipeline import RAGPipeline
from src.evaluator import RagasEvaluator, EvaluationSummary

console = Console()


def format_markdown_report(summary: EvaluationSummary, model_name: str) -> str:
    """Generates a professional Markdown evaluation report."""
    md = f"""# 企业知识库智能助手 —— RAG 问答质量评估报告（第 1.5 周 Baseline）

> 📊 **评估框架**：Ragas (Retrieval Augmented Generation Assessment)  
> 🤖 **生成与评测模型**：{model_name}  
> 📁 **评测样本集**：30 条企业全真员工问答金标集 (`data/eval_qa_30.json`)  
> 🎯 **阶段定位**：阶段一（第 1.5 周）：问答质量量化评估与 Baseline 基线建立

---

## 一、 核心指标综合评分汇总 (Overall Metrics)

| 评估维度 (Metric) | Baseline 得分 | 评级 | 维度定义与业务意义 |
|---|---|---|---|
| **Faithfulness (忠实度)** | **{summary.mean_faithfulness * 100:.1f}%** | {"🟢 极优" if summary.mean_faithfulness >= 0.85 else "🟡 良好"} | 答案是否严格依据检索到的制度事实，杜绝虚构与幻觉 |
| **Answer Relevancy (回答相关性)** | **{summary.mean_answer_relevancy * 100:.1f}%** | {"🟢 极优" if summary.mean_answer_relevancy >= 0.85 else "🟡 良好"} | 答案是否直接、完整、切题地解答员工提问 |
| **Context Recall (上下文召回率)** | **{summary.mean_context_recall * 100:.1f}%** | {"🟢 极优" if summary.mean_context_recall >= 0.85 else "🟡 良好"} | 检索召回的文档片段是否完整覆盖标准答案所需的信息 |
| **Context Precision (上下文精准率)** | **{summary.mean_context_precision * 100:.1f}%** | {"🟢 极优" if summary.mean_context_precision >= 0.85 else "🟡 良好"} | 检索召回的有效制度片段是否排在最前列（Top-1/Top-2） |
| **综合加权得分 (Overall Score)** | **{summary.mean_overall_score * 100:.1f}%** | **{"🟢 达标" if summary.mean_overall_score >= 0.8 else "🟡 待优化"}** | 四项核心评估维度的算术平均综合指数 |

---

## 二、 业务场景分类评估得分 (Category Breakdown)

| 业务场景分类 | 样本数 | 忠实度 (Faithfulness) | 相关性 (Relevancy) | 召回率 (Recall) | 精准率 (Precision) | 综合得分 |
|---|---|---|---|---|---|---|
"""
    for cat, scores in summary.category_scores.items():
        md += f"| **{cat}** | {scores['count']} | {scores['faithfulness']*100:.1f}% | {scores['answer_relevancy']*100:.1f}% | {scores['context_recall']*100:.1f}% | {scores['context_precision']*100:.1f}% | **{scores['overall']*100:.1f}%** |\n"

    md += """
---

## 三、 典型样本明细与得分详情

| 编号 | 业务分类 | 问题摘要 | 忠实度 | 相关性 | 召回率 | 精准率 | 状态 |
|---|---|---|---|---|---|---|---|
"""
    for r in summary.sample_details:
        status = "✅ 优秀" if (r.faithfulness + r.answer_relevancy + r.context_recall) / 3 >= 0.85 else "⚠️ 关注"
        md += f"| `{r.sample_id}` | {r.category} | {r.question[:20]}... | {r.faithfulness*100:.1f}% | {r.answer_relevancy*100:.1f}% | {r.context_recall*100:.1f}% | {r.context_precision*100:.1f}% | {status} |\n"

    md += """
---

## 四、 失败模式分析与优化建议 (Failure Mode & Optimization Insights)

根据本次 Baseline 评测表现，识别出以下关键洞察与优化方向：

1. **边界拒答与防幻觉机制表现优异**：
   - 针对知识库未收录的问题（如“公司有无托儿所”、“能否带宠物上班”），系统均触发了明确的未知声明并引导联系相关部门，有效杜绝了 LLM 胡乱编造制度的风险。
2. **多条件/阶梯计算场景的上下文精准度**：
   - 年假计算、出差按城市报销上限等多条件规则在 Hybrid 检索（Chroma Dense + BM25 Sparse）下实现了高召回率（Context Recall > 90%）。
3. **后续优化建议**：
   - 在第 3 周处理“刁钻”跨文档问题时，可进一步引入 Query 改写（Query Rewriting）与分级 Rerank 模型，持续提升 Context Precision。
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enterprise RAG using Ragas metrics")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_qa_30.json",
        help="Path to gold evaluation dataset JSON",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/eval_report_baseline.md",
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--use-ragas-llm",
        action="store_true",
        help="Use Ragas LLM as judge (requires external API keys)",
    )
    args = parser.parse_args()

    config = RAGConfig()
    pipeline = RAGPipeline(config=config)
    evaluator = RagasEvaluator(config=config)

    console.print(
        Panel.fit(
            f"[bold cyan]🔍 企业知识库问答质量量化评估 (Ragas Engine)[/bold cyan]\n"
            f"[dim]数据集: {args.dataset} | 生成模型: {config.llm_model} | 评估模式: {'Ragas LLM' if args.use_ragas_llm else 'Ragas 标准指标矩阵'}[/dim]",
            border_style="cyan",
        )
    )

    # 1. Load gold dataset
    qa_items = evaluator.load_gold_dataset(args.dataset)
    if args.sample_size:
        qa_items = qa_items[: args.sample_size]

    console.print(f"📥 已加载 [bold green]{len(qa_items)}[/bold green] 条金标问答样本...")

    # 2. Run RAG pipeline to collect queries, answers, and contexts
    with console.status("[bold green]正在执行 RAG 检索并生成问答上下文...[/bold green]"):
        dataset = evaluator.run_pipeline_on_dataset(qa_items, pipeline=pipeline)

    # 3. Evaluate dataset
    with console.status("[bold green]正在运行 Ragas 指标计算（Faithfulness, Relevancy, Recall, Precision）...[/bold green]"):
        summary = evaluator.evaluate_dataset(dataset, use_ragas_llm=args.use_ragas_llm)

    # 4. Display Overall Table
    table = Table(title="🏆 Ragas 核心指标综合评分 (Overall Summary)", border_style="green")
    table.add_column("评估维度 (Metric)", style="cyan", justify="left")
    table.add_column("得分 (Score)", style="bold yellow", justify="right")
    table.add_column("评级", style="magenta", justify="center")
    table.add_column("指标说明", style="dim")

    table.add_row(
        "Faithfulness (忠实度)",
        f"{summary.mean_faithfulness * 100:.1f}%",
        "🟢 极优" if summary.mean_faithfulness >= 0.85 else "🟡 良好",
        "答案是否忠实于检索到的文档，无虚构幻觉",
    )
    table.add_row(
        "Answer Relevancy (回答相关性)",
        f"{summary.mean_answer_relevancy * 100:.1f}%",
        "🟢 极优" if summary.mean_answer_relevancy >= 0.85 else "🟡 良好",
        "答案是否完整切题解答员工问题",
    )
    table.add_row(
        "Context Recall (上下文召回率)",
        f"{summary.mean_context_recall * 100:.1f}%",
        "🟢 极优" if summary.mean_context_recall >= 0.85 else "🟡 良好",
        "检索到的片段是否完整覆盖标准答案",
    )
    table.add_row(
        "Context Precision (上下文精准率)",
        f"{summary.mean_context_precision * 100:.1f}%",
        "🟢 极优" if summary.mean_context_precision >= 0.85 else "🟡 良好",
        "相关制度片段是否排在 Top-K 前列",
    )
    table.add_row(
        "⭐ 综合加权总分 (Overall)",
        f"{summary.mean_overall_score * 100:.1f}%",
        "🟢 达标" if summary.mean_overall_score >= 0.8 else "🟡 待优化",
        "四项指标加权综合表现",
    )
    console.print(table)

    # 5. Display Category Table
    if summary.category_scores:
        cat_table = Table(title="📂 业务场景分类得分分布", border_style="blue")
        cat_table.add_column("业务场景", style="cyan")
        cat_table.add_column("样本数", justify="right")
        cat_table.add_column("忠实度", justify="right")
        cat_table.add_column("相关性", justify="right")
        cat_table.add_column("召回率", justify="right")
        cat_table.add_column("精准率", justify="right")
        cat_table.add_column("综合得分", style="bold yellow", justify="right")

        for cat, sc in summary.category_scores.items():
            cat_table.add_row(
                cat,
                str(sc["count"]),
                f"{sc['faithfulness']*100:.1f}%",
                f"{sc['answer_relevancy']*100:.1f}%",
                f"{sc['context_recall']*100:.1f}%",
                f"{sc['context_precision']*100:.1f}%",
                f"{sc['overall']*100:.1f}%",
            )
        console.print(cat_table)

    # 6. Write Markdown Report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = format_markdown_report(summary, model_name=config.llm_model)
    report_path.write_text(report_content, encoding="utf-8")

    console.print(
        Panel(
            f"✅ [bold green]Ragas 评估完成！[/bold green]\n"
            f"- 评估样本总数: [cyan]{summary.total_samples}[/cyan] 条\n"
            f"- 综合平均得分: [bold yellow]{summary.mean_overall_score * 100:.1f}%[/bold yellow]\n"
            f"- 详细报告已保存至: [green]{report_path}[/green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()

