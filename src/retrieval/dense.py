"""
Dense retriever — embed query, vector search, optional metadata filter.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.embeddings.embedder import BaseEmbedder
from src.retrieval.filters import apply_date_filter, build_filter


class DenseRetriever:
    """
    Retriever A: dense vector search with metadata filtering.

    Fast and controllable. Metadata filters are applied at the vectorstore
    level (Pinecone: server-side; FAISS: Python-side after over-fetch).
    """

    def __init__(self, vectorstore: Any, embedder: BaseEmbedder, top_k: int = 5) -> None:
        self._store = vectorstore
        self._embedder = embedder
        self.top_k = top_k

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
        Embed *query* and return the top-k matching chunks.

        Args:
            query: Natural language query.
            company_name: Filter to a specific company.
            doc_source: "transcript" | "sec_filing" | None (both).
            document_type: e.g. "10-K", "Transcript".
            date_from / date_to: Optional date range filter.

        Returns:
            List of chunk metadata dicts, sorted by relevance.
        """
        # Use embed_query() so E5 models get the "query: " prefix at retrieval time.
        # Falls back to embed_one() for embedders that don't implement embed_query().
        embed_fn = getattr(self._embedder, "embed_query", self._embedder.embed_one)
        query_vec = embed_fn(query)
        filters = build_filter(company_name, doc_source, document_type)

        results = self._store.query_with_filter(
            query_vector=query_vec,
            k=self.top_k,
            filters=filters if filters else None,
        )

        return apply_date_filter(results, date_from, date_to)
