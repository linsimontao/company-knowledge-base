"""Configuration module for Enterprise Knowledge Base RAG system."""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_DOCS_DIR = DEFAULT_DATA_DIR / "documents"
DEFAULT_CHROMA_DIR = DEFAULT_DATA_DIR / "chroma_db"


class RAGConfig(BaseModel):
    """RAG pipeline configurations."""

    # Storage paths
    docs_dir: Path = Field(default=DEFAULT_DOCS_DIR)
    chroma_dir: Path = Field(default=DEFAULT_CHROMA_DIR)
    collection_name: str = Field(default="company_knowledge_base")

    # Chunking parameters
    chunk_size: int = Field(
        default=650,
        description="Target maximum character count per chunk (optimized for Chinese policy clauses and tables)",
    )
    chunk_overlap: int = Field(
        default=120,
        description="Character overlap between consecutive chunks to preserve boundary context",
    )

    # Retrieval parameters
    top_k: int = Field(default=5, description="Number of relevant chunks to retrieve")
    score_threshold: float = Field(
        default=0.0, description="Minimum similarity score threshold (0.0 to 1.0)"
    )

    # Embedding provider: 'chroma_default', 'openai', 'gemini', or 'mock'
    embedding_provider: str = Field(
        default=os.getenv("EMBEDDING_PROVIDER", "chroma_default")
    )
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # LLM provider: 'openai', 'gemini', or 'mock'
    llm_provider: str = Field(default=os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = Field(default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = Field(default=0.0)

    # API Keys
    openai_api_key: str | None = Field(default=os.getenv("OPENAI_API_KEY"))
    openai_base_url: str | None = Field(default=os.getenv("OPENAI_BASE_URL"))
    gemini_api_key: str | None = Field(default=os.getenv("GEMINI_API_KEY"))


default_config = RAGConfig()
