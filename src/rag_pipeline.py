"""End-to-End Enterprise RAG Pipeline orchestrator."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table

from src.config import RAGConfig, default_config
from src.chunker import MarkdownChunker, DocumentChunk
from src.vector_store import ChromaVectorStore
from src.retriever import DocumentRetriever, RetrievedDoc
from src.generator import LLMGenerator, GeneratedAnswer

console = Console()


class RAGPipeline:
    """Orchestrates document loading, chunking, indexing, retrieval, and generation."""

    def __init__(self, config: RAGConfig = default_config):
        self.config = config
        self.chunker = MarkdownChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.vector_store = ChromaVectorStore(config)
        self.retriever = DocumentRetriever(self.vector_store, config)
        self.generator = LLMGenerator(config)

    def build_knowledge_base(
        self, docs_dir: Optional[str | Path] = None, reset: bool = True
    ) -> Dict[str, Any]:
        """Ingests and indexes all Markdown documents from the documents directory."""
        target_dir = Path(docs_dir or self.config.docs_dir)
        if not target_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {target_dir}")

        if reset:
            self.vector_store.reset_collection()

        doc_files = sorted(list(target_dir.glob("*.md")))
        if not doc_files:
            raise ValueError(f"No Markdown files found in {target_dir}")

        all_chunks: List[DocumentChunk] = []
        file_stats: List[Dict[str, Any]] = []

        for fpath in doc_files:
            chunks = self.chunker.chunk_file(fpath)
            all_chunks.extend(chunks)
            file_stats.append(
                {
                    "file": fpath.name,
                    "chunks": len(chunks),
                    "chars": sum(c.metadata.get("char_count", 0) for c in chunks),
                }
            )

        # Upsert all chunks into ChromaDB
        added_count = self.vector_store.add_chunks(all_chunks)

        return {
            "total_files": len(doc_files),
            "total_chunks": added_count,
            "file_stats": file_stats,
            "collection_count": self.vector_store.count(),
        }

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> GeneratedAnswer:
        """Executes full RAG query: retrieve relevant chunks -> generate grounded response."""
        retrieved_docs = self.retriever.retrieve(
            query=query_text,
            top_k=top_k,
            where=where,
        )
        answer = self.generator.generate(
            query=query_text,
            retrieved_docs=retrieved_docs,
        )
        return answer

    def get_stats(self) -> Dict[str, Any]:
        """Returns runtime vector store and configuration statistics."""
        return {
            "collection_name": self.config.collection_name,
            "total_indexed_chunks": self.vector_store.count(),
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "embedding_provider": self.config.embedding_provider,
            "llm_provider": self.config.llm_provider,
            "top_k": self.config.top_k,
        }

