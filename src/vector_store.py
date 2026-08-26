"""Vector store manager wrapping ChromaDB."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from src.config import RAGConfig, default_config
from src.chunker import DocumentChunk
from src.embeddings import get_embedding_function


class ChromaVectorStore:
    """Manages ChromaDB persistent collections and index operations."""

    def __init__(self, config: RAGConfig = default_config):
        self.config = config
        self.chroma_dir = Path(config.chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.embedding_function = get_embedding_function(config)
        self.collection_name = config.collection_name

        self._collection: Optional[Collection] = None

    @property
    def collection(self) -> Collection:
        """Lazy-loads or retrieves the Chroma collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 50) -> int:
        """Adds a list of document chunks to the collection in batches."""
        if not chunks:
            return 0

        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_added += len(batch)

        return total_added

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Queries the vector store for most similar chunks."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def count(self) -> int:
        """Returns the total number of documents/chunks in the collection."""
        return self.collection.count()

    def reset_collection(self) -> None:
        """Deletes the collection if it exists and creates an empty one."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

