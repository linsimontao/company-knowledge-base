#!/usr/bin/env python3
"""Build and index knowledge base documents into ChromaDB."""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

from src.config import RAGConfig
from src.rag_pipeline import RAGPipeline

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Build Enterprise Knowledge Base ChromaDB Index")
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="data/documents",
        help="Path to directory containing Markdown documents",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target maximum character count per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Character overlap between consecutive chunks",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not reset/wipe the existing collection before indexing",
    )
    args = parser.parse_args()

    config = RAGConfig(
        docs_dir=Path(args.docs_dir),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    console.print(
        Panel.fit(
            f"[bold cyan]星云智联企业知识库 —— 向量数据库索引构建器[/bold cyan]\n"
            f"[dim]文档目录: {config.docs_dir} | Chunk Size: {config.chunk_size} | Overlap: {config.chunk_overlap}[/dim]",
            border_style="cyan",
        )
    )

    pipeline = RAGPipeline(config=config)

    with console.status("[bold green]正在读取并切块 Markdown 文档，写入 Chroma 向量库...[/bold green]"):
        result = pipeline.build_knowledge_base(
            docs_dir=config.docs_dir,
            reset=not args.no_reset,
        )

    # Print results table
    table = Table(title="📚 文档分块与向量索引统计", border_style="blue")
    table.add_column("序号", justify="right", style="cyan")
    table.add_column("文档名称", style="green")
    table.add_column("切块数量 (Chunks)", justify="right", style="magenta")
    table.add_column("总字符数", justify="right", style="yellow")

    for idx, item in enumerate(result["file_stats"], 1):
        table.add_row(
            str(idx),
            item["file"],
            str(item["chunks"]),
            f"{item['chars']:,}",
        )

    console.print(table)
    console.print(
        Panel(
            f"✅ [bold green]知识库索引构建完成！[/bold green]\n"
            f"- 处理文档总数: [cyan]{result['total_files']}[/cyan] 篇\n"
            f"- 生成向量切块: [magenta]{result['total_chunks']}[/magenta] 个\n"
            f"- 向量库持久化目录: [yellow]{config.chroma_dir}[/yellow]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()

