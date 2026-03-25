"""
BM25 sparse retriever.

Uses keyword-based BM25 scoring for retrieval. Particularly strong for
exact financial terms (ticker symbols, metric names like "EBITDA",
document types like "10-K") where dense embeddings may underperform.

Requires: rank-bm25>=0.2.2
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.filters import apply_date_filter, build_filter


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """
    BM25 sparse retriever.

    Builds a BM25Okapi index from pre-loaded chunk records at init time.
    At query time, tokenizes the query, scores against the index, applies
    metadata filters, and returns top-k results.

    Args:
        chunks: Pre-loaded list of chunk metadata dicts (must contain "text").
        top_k: Number of results to return.
    """

    def __init__(self, chunks: list[dict[str, Any]], top_k: int = 5) -> None:
        self.top_k = top_k
        self._chunks = chunks
        corpus = [_tokenize(c.get("text", "")) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

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
        Retrieve top-k chunks by BM25 score.

        Supports the same filter interface as DenseRetriever for
        interchangeability in the eval pipeline.
        """
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Pair scores with chunks and sort
        scored = list(zip(scores, self._chunks))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply metadata filters
        filters = build_filter(company_name, doc_source, document_type)
        results: list[dict[str, Any]] = []
        for score, chunk in scored:
            if score <= 0:
                continue
            if filters:
                skip = False
                for key, val in filters.items():
                    if chunk.get(key) != val:
                        skip = True
                        break
                if skip:
                    continue
            result = dict(chunk)
            result["_score"] = float(score)
            results.append(result)

        results = apply_date_filter(results, date_from, date_to)
        return results[: self.top_k]
