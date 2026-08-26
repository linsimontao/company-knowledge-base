"""Hybrid retrieval engine combining ChromaDB vector search and BM25 sparse keyword search."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.config import RAGConfig, default_config
from src.vector_store import ChromaVectorStore

try:
    import jieba
    from rank_bm25 import BM25Okapi
except ImportError:
    jieba = None
    BM25Okapi = None


class RetrievedDoc(BaseModel):
    """Represents a retrieved document chunk with relevance metrics."""

    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    header_path: str
    distance: float = Field(default=0.0, description="Cosine distance (lower is closer)")
    similarity_score: float = Field(
        description="Hybrid relevance score between 0.0 and 1.0 (higher is more similar)"
    )
    rank: int = Field(description="Rank position (1-indexed)")


class DocumentRetriever:
    """Retrieves top-k relevant chunks from ChromaDB with optional BM25 hybrid ranking."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        config: RAGConfig = default_config,
    ):
        self.config = config
        self.vector_store = vector_store or ChromaVectorStore(config)
        self._bm25 = None
        self._corpus_ids = []
        self._corpus_docs = []
        self._corpus_metas = []

    def _init_bm25_if_needed(self):
        """Builds in-memory BM25 index over all chunks in Chroma."""
        if not BM25Okapi or not jieba:
            return

        if self._bm25 is not None:
            return

        all_records = self.vector_store.collection.get(include=["documents", "metadatas"])
        if not all_records or not all_records.get("documents"):
            return

        self._corpus_ids = all_records["ids"]
        self._corpus_docs = all_records["documents"]
        self._corpus_metas = all_records["metadatas"] or [{}] * len(self._corpus_docs)

        tokenized_corpus = [
            list(jieba.cut_for_search(doc.lower())) for doc in self._corpus_docs
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[RetrievedDoc]:
        """Retrieves and scores top-k relevant chunks using Hybrid Fusion (Chroma Dense + BM25 Sparse)."""
        k = top_k or self.config.top_k
        fetch_k = min(20, max(k * 3, 10))

        # 1. Chroma Dense Vector Search
        raw_results = self.vector_store.query(query_text=query, top_k=fetch_k, where=where)

        dense_ranks: Dict[str, Dict[str, Any]] = {}
        if raw_results and raw_results.get("documents") and raw_results["documents"][0]:
            documents = raw_results["documents"][0]
            metadatas = (
                raw_results["metadatas"][0]
                if raw_results.get("metadatas")
                else [{}] * len(documents)
            )
            distances = (
                raw_results["distances"][0]
                if raw_results.get("distances")
                else [0.0] * len(documents)
            )
            ids = (
                raw_results["ids"][0]
                if raw_results.get("ids")
                else [f"chunk_{i}" for i in range(len(documents))]
            )

            for rank_idx, (chunk_id, doc_text, meta, dist) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                sim = max(0.0, min(1.0, 1.0 - float(dist)))
                dense_ranks[chunk_id] = {
                    "rank": rank_idx + 1,
                    "sim": sim,
                    "text": doc_text,
                    "meta": meta,
                    "dist": dist,
                }

        # 2. BM25 Sparse Keyword Search
        self._init_bm25_if_needed()
        sparse_ranks: Dict[str, Dict[str, Any]] = {}
        if self._bm25 and jieba:
            q_tokens = list(jieba.cut_for_search(query.lower()))
            bm25_scores = self._bm25.get_scores(q_tokens)
            top_bm25_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[:fetch_k]

            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            for rank_idx, idx in enumerate(top_bm25_indices):
                if bm25_scores[idx] <= 0:
                    continue
                chunk_id = self._corpus_ids[idx]
                sparse_ranks[chunk_id] = {
                    "rank": rank_idx + 1,
                    "score": bm25_scores[idx] / max_bm25,
                    "text": self._corpus_docs[idx],
                    "meta": self._corpus_metas[idx],
                }

        # 3. Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(dense_ranks.keys()).union(sparse_ranks.keys())
        rrf_constant = 60.0
        combined_scores: List[Dict[str, Any]] = []

        for cid in all_candidate_ids:
            score = 0.0
            doc_text = ""
            doc_meta = {}
            dist = 1.0

            if cid in dense_ranks:
                d_info = dense_ranks[cid]
                score += dense_weight * (1.0 / (rrf_constant + d_info["rank"]))
                doc_text = d_info["text"]
                doc_meta = d_info["meta"]
                dist = d_info["dist"]

            if cid in sparse_ranks:
                s_info = sparse_ranks[cid]
                score += sparse_weight * (1.0 / (rrf_constant + s_info["rank"]))
                if not doc_text:
                    doc_text = s_info["text"]
                    doc_meta = s_info["meta"]

            combined_scores.append(
                {
                    "chunk_id": cid,
                    "text": doc_text,
                    "meta": doc_meta,
                    "dist": dist,
                    "score": score,
                }
            )

        # Sort by combined RRF score descending
        combined_scores.sort(key=lambda x: x["score"], reverse=True)

        # Scale scores to friendly 0.0 - 1.0 range
        max_score = combined_scores[0]["score"] if combined_scores else 1.0
        if max_score <= 0:
            max_score = 1.0

        docs: List[RetrievedDoc] = []
        for idx, item in enumerate(combined_scores[:k]):
            norm_sim = min(1.0, item["score"] / max_score)
            if norm_sim < self.config.score_threshold:
                continue

            meta = item["meta"] or {}
            docs.append(
                RetrievedDoc(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    source=meta.get("source", "unknown"),
                    title=meta.get("title", "unknown"),
                    section=meta.get("section", "unknown"),
                    header_path=meta.get("header_path", "unknown"),
                    distance=round(float(item["dist"]), 4),
                    similarity_score=round(norm_sim, 4),
                    rank=idx + 1,
                )
            )

        return docs

