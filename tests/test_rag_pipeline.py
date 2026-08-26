"""Comprehensive automated tests for Enterprise RAG Pipeline."""

import pytest
from pathlib import Path
from src.config import RAGConfig
from src.chunker import MarkdownChunker
from src.vector_store import ChromaVectorStore
from src.retriever import DocumentRetriever
from src.generator import LLMGenerator
from src.rag_pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "documents"


def test_documents_exist_and_complete():
    """Verify that all 15 company knowledge base documents exist and have content."""
    assert DOCS_DIR.exists(), f"Docs directory {DOCS_DIR} does not exist"
    doc_files = list(DOCS_DIR.glob("*.md"))
    assert len(doc_files) >= 15, f"Expected at least 15 documents, found {len(doc_files)}"

    for doc_file in doc_files:
        content = doc_file.read_text(encoding="utf-8").strip()
        assert len(content) > 200, f"Document {doc_file.name} is too short ({len(content)} chars)"
        assert content.startswith("# "), f"Document {doc_file.name} must start with H1 header"


def test_markdown_chunker():
    """Verify chunking logic, header parsing, and overlap handling."""
    chunker = MarkdownChunker(chunk_size=400, chunk_overlap=80)
    test_doc = DOCS_DIR / "01_员工考勤与请假管理制度.md"
    chunks = chunker.chunk_file(test_doc)

    assert len(chunks) > 0, "No chunks generated"
    for c in chunks:
        assert c.chunk_id.startswith("01_员工考勤与请假管理制度.md#chunk_")
        assert "source" in c.metadata
        assert "section" in c.metadata
        assert "header_path" in c.metadata
        assert len(c.text) > 0


def test_vector_store_and_retrieval():
    """Verify Chroma vector store indexing and retrieval recall for key queries."""
    test_config = RAGConfig(
        docs_dir=DOCS_DIR,
        chroma_dir=PROJECT_ROOT / "data" / "test_chroma_db",
        collection_name="test_knowledge_base",
        chunk_size=450,
        chunk_overlap=80,
        embedding_provider="chroma_default",
    )

    pipeline = RAGPipeline(config=test_config)
    build_result = pipeline.build_knowledge_base(reset=True)

    assert build_result["total_files"] >= 15
    assert build_result["total_chunks"] > 0
    assert pipeline.vector_store.count() == build_result["total_chunks"]

    # Test Query 1: Leave policy / Annual leave
    docs_leave = pipeline.retriever.retrieve("年假一年有多少天", top_k=3)
    assert len(docs_leave) > 0
    assert any("请假" in d.source or "考勤" in d.source for d in docs_leave)

    # Test Query 2: Expense reimbursement documents
    docs_expense = pipeline.retriever.retrieve("报销需要哪些材料", top_k=3)
    assert len(docs_expense) > 0
    assert any("报销" in d.source for d in docs_expense)

    # Test Query 3: VPN access request
    docs_vpn = pipeline.retriever.retrieve("怎么申请VPN权限", top_k=3)
    assert len(docs_vpn) > 0
    assert any("IT" in d.source or "VPN" in d.text for d in docs_vpn)


def test_end_to_end_rag_query():
    """Verify full RAG pipeline query execution and grounded answer generation."""
    test_config = RAGConfig(
        docs_dir=DOCS_DIR,
        chroma_dir=PROJECT_ROOT / "data" / "test_chroma_db",
        collection_name="test_knowledge_base",
        embedding_provider="chroma_default",
        llm_provider="mock",
    )

    pipeline = RAGPipeline(config=test_config)
    answer = pipeline.query("年假一年有多少天")

    assert answer.query == "年假一年有多少天"
    assert len(answer.answer) > 0
    assert len(answer.sources) > 0
    assert len(answer.retrieved_docs) > 0

