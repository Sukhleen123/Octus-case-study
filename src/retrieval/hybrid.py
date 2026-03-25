"""
Hybrid retriever: dense + BM25 with Reciprocal Rank Fusion.

Combines semantic similarity (dense embeddings) with keyword matching
(BM25) for robust retrieval across both conceptual and exact-match queries.

RRF is parameter-free (aside from k=60), requires no training data,
and works with heterogeneous score distributions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

_RRF_K = 60  # Standard RRF constant


class HybridRetriever:
    """
    Hybrid dense + BM25 retriever with Reciprocal Rank Fusion.

    Runs both retrievers, fuses results via RRF, deduplicates by
    chunk_id, and returns the top-k results.

    Args:
        dense_retriever: A DenseRetriever or DenseMMRRetriever instance.
        bm25_retriever: A BM25Retriever instance.
        top_k: Number of results to return.
        dense_weight: Weight for dense scores in fusion (0.0-1.0).
    """

    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: Any,
        top_k: int = 5,
        dense_weight: float = 0.6,
    ) -> None:
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.bm25_weight = 1.0 - dense_weight

    def retrieve(
        self,
        query: str,
        company_name: str | None = None,
        doc_source: str | None = None,
        document_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve chunks using both dense and BM25, fused via RRF.
        """
        kwargs = dict(
            query=query,
            company_name=company_name,
            doc_source=doc_source,
            document_type=document_type,
            date_from=date_from,
            date_to=date_to,
        )

        # Over-fetch from both retrievers
        fetch_k = self.top_k * 2
        orig_dense_k = getattr(self._dense, "top_k", self.top_k)
        orig_bm25_k = getattr(self._bm25, "top_k", self.top_k)

        self._dense.top_k = fetch_k
        self._bm25.top_k = fetch_k

        try:
            dense_results = self._dense.retrieve(**kwargs)
            bm25_results = self._bm25.retrieve(**kwargs)
        finally:
            self._dense.top_k = orig_dense_k
            self._bm25.top_k = orig_bm25_k

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(dense_results):
            cid = result.get("chunk_id", str(rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.dense_weight / (_RRF_K + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = result

        for rank, result in enumerate(bm25_results):
            cid = result.get("chunk_id", str(rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.bm25_weight / (_RRF_K + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = result

        # Sort by fused RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results: list[dict[str, Any]] = []
        for cid, score in ranked[: self.top_k]:
            result = dict(chunk_data[cid])
            result["_score"] = score
            result["_rrf_score"] = score
            results.append(result)

        return results
