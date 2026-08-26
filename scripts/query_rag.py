#!/usr/bin/env python3
"""Interactive query and test runner for Enterprise Knowledge Base RAG."""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.config import RAGConfig
from src.rag_pipeline import RAGPipeline

console = Console()

TEST_QUESTIONS = [
    "年假一年有多少天",
    "报销需要哪些材料",
    "怎么申请VPN权限",
    "试用期一般是多久？转正有什么要求？",
    "每周可以申请几天远程办公？",
    "加班到晚上有餐补和打车报销吗？",
    "公司有提供内部托儿所吗？",  # Edge case: not in docs
]


def display_query_result(query_text: str, result):
    """Renders formatted query results and retrieved chunks using Rich."""
    console.print()
    console.print(Panel(f"[bold yellow]❓ 员工提问：[/bold yellow] [bold white]{query_text}[/bold white]", border_style="yellow"))

    # Table of retrieved documents
    if result.retrieved_docs:
        table = Table(title="🔍 知识库检索命中片段 (Top-K)", border_style="blue", show_lines=True)
        table.add_column("Rank", justify="center", style="cyan", width=6)
        table.add_column("来源文档", style="green", width=28)
        table.add_column("章节路径", style="magenta", width=30)
        table.add_column("相似度", justify="right", style="yellow", width=10)
        table.add_column("内容摘要", style="dim", width=45)

        for doc in result.retrieved_docs:
            snippet = doc.text.replace("\n", " ")[:80] + ("..." if len(doc.text) > 80 else "")
            table.add_row(
                str(doc.rank),
                doc.source,
                doc.section,
                f"{doc.similarity_score:.3f}",
                snippet,
            )
        console.print(table)
    else:
        console.print("[dim italic]（未检索到相关文档片段）[/dim italic]")

    # Render generated answer
    console.print(
        Panel(
            Markdown(result.answer),
            title=f"🤖 [bold green]智能助手回答 (模型: {result.model_used})[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


def run_batch_tests(pipeline: RAGPipeline):
    """Runs standard test benchmark suite."""
    console.print(
        Panel.fit(
            "[bold cyan]🚀 开始执行企业知识库问答自动化测试集[/bold cyan]",
            border_style="cyan",
        )
    )

    for idx, q in enumerate(TEST_QUESTIONS, 1):
        console.print(f"\n[bold cyan]▶ 测试用例 {idx}/{len(TEST_QUESTIONS)}[/bold cyan]")
        result = pipeline.query(q)
        display_query_result(q, result)


def main():
    parser = argparse.ArgumentParser(description="Query Enterprise Knowledge Base RAG")
    parser.add_argument("--query", "-q", type=str, help="Single query text to ask")
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Run standard test questions benchmark",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve",
    )
    args = parser.parse_args()

    config = RAGConfig(top_k=args.top_k)
    pipeline = RAGPipeline(config=config)

    # Check if index exists
    if pipeline.vector_store.count() == 0:
        console.print("[yellow]⚠️ 向量数据库为空，正在自动构建知识库索引...[/yellow]")
        pipeline.build_knowledge_base()

    if args.test_all:
        run_batch_tests(pipeline)
        return

    if args.query:
        result = pipeline.query(args.query)
        display_query_result(args.query, result)
        return

    # Interactive mode
    console.print(
        Panel.fit(
            "[bold green]🏢 星云智联企业知识库智能问答系统 (CLI 交互终端)[/bold green]\n"
            "[dim]输入您的问题并回车进行提问，输入 'exit' 或 'quit' 退出系统[/dim]",
            border_style="green",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold cyan]请输入问题 > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[green]感谢使用，再见！[/green]")
                break
            result = pipeline.query(user_input)
            display_query_result(user_input, result)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[green]已退出问答系统。[/green]")
            break


if __name__ == "__main__":
    main()

