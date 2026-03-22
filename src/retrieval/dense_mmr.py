"""
Dense + MMR retriever.

MMR (Maximal Marginal Relevance) re-ranks retrieved candidates to reduce
redundancy while maintaining relevance. Useful when multiple chunks from the
same section/document would otherwise dominate the results.

See docs/decisions.md for tradeoff discussion.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from src.embeddings.embedder import BaseEmbedder
from src.retrieval.filters import apply_date_filter, build_filter


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return sum(x * y for x, y in zip(a, b))


def _mmr_rerank(
    query_vec: list[float],
    candidates: list[dict[str, Any]],
    candidate_vecs: list[list[float]],
    k: int,
    lambda_: float,
) -> list[dict[str, Any]]:
    """
    MMR selection from candidates.

    lambda_=1.0 → pure relevance (same as dense)
    lambda_=0.0 → pure diversity
    """
    selected_indices: list[int] = []
    selected_vecs: list[list[float]] = []

    scores = [_cosine(query_vec, v) for v in candidate_vecs]

    for _ in range(min(k, len(candidates))):
        best_idx = -1
        best_mmr = float("-inf")

        for i in range(len(candidates)):
            if i in selected_indices:
                continue
            relevance = scores[i]
            if selected_vecs:
                redundancy = max(_cosine(candidate_vecs[i], sv) for sv in selected_vecs)
            else:
                redundancy = 0.0
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        selected_vecs.append(candidate_vecs[best_idx])

    return [candidates[i] for i in selected_indices]


class DenseMMRRetriever:
    """
    Retriever B: dense search + MMR re-ranking.

    Over-fetches from the vectorstore, then applies MMR to select diverse results.
    Requires storing query-time embeddings of candidates — not supported by
    Pinecone's query response (scores only). For Pinecone we re-embed the
    retrieved chunk texts; for FAISS candidates already have score approximations.
    """

    def __init__(
        self,
        vectorstore: Any,
        embedder: BaseEmbedder,
        top_k: int = 5,
        mmr_lambda: float = 0.5,
        overfetch_factor: int = 4,
    ) -> None:
        self._store = vectorstore
        self._embedder = embedder
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.overfetch_factor = overfetch_factor

    def retrieve(
        self,
        query: str,
        company_name: str | None = None,
        doc_source: str | None = None,
        document_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        embed_fn = getattr(self._embedder, "embed_query", self._embedder.embed_one)
        query_vec = embed_fn(query)
        filters = build_filter(company_name, doc_source, document_type)

        # Over-fetch for MMR candidate pool
        fetch_k = self.top_k * self.overfetch_factor
        candidates = self._store.query_with_filter(
            query_vector=query_vec,
            k=fetch_k,
            filters=filters if filters else None,
        )
        candidates = apply_date_filter(candidates, date_from, date_to)

        if not candidates:
            return []

        # Re-embed candidate texts for MMR similarity computation
        texts = [c.get("text", "") for c in candidates]
        candidate_vecs = self._embedder.embed(texts)

        return _mmr_rerank(
            query_vec, candidates, candidate_vecs, self.top_k, self.mmr_lambda
        )
