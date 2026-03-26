"""
MultiStoreRetriever: routes queries across two Pinecone namespaces.

Stores:
  transcript_retriever — transcripts chunked with RecursiveChunker
  sec_retriever        — SEC filings chunked with SECSectionChunker

Routing (by doc_source + query signal):
  doc_source="transcript"  → transcript store only
  doc_source="sec_filing"  → SEC store only; section queries prioritized
  default (unknown/both)   → both stores; section queries lead with SEC

Boilerplate filter: Cautionary Note / Safe Harbor disclaimer chunks are
de-prioritized to the back of results — returned only if fewer than top_k
clean chunks are available.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SECTION_QUERY_KEYWORDS = frozenset({
    "risk factor", "item 1a", "md&a", "management discussion",
    "balance sheet", "income statement", "cash flow", "notes to financial",
    "financial statements", "item 7", "item 2",
})

_BOILERPLATE_HEADERS = (
    "cautionary note on forward-looking statements",
    "cautionary note regarding forward-looking statements",
    "safe harbor statement under the private securities litigation reform act",
    "note regarding forward-looking statements",
)


class MultiStoreRetriever:
    """Routes retrieval across transcript and SEC filing stores."""

    NAME = "multi_store"

    def __init__(
        self,
        transcript_retriever: Any,
        sec_retriever: Any,
        top_k: int = 10,
    ) -> None:
        self._transcript = transcript_retriever
        self._sec = sec_retriever
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        doc_source: str | None = None,
        **kwargs,
    ) -> list[dict]:
        targets = self._route(query, doc_source)
        all_chunks: list[dict] = []
        seen: set[str] = set()

        for retriever in targets:
            try:
                chunks = retriever.retrieve(query, doc_source=doc_source, **kwargs)
            except TypeError:
                chunks = retriever.retrieve(query, **kwargs)
            for c in chunks:
                cid = c.get("chunk_id", str(id(c)))
                if cid not in seen:
                    seen.add(cid)
                    all_chunks.append(c)

        all_chunks.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

        clean = [c for c in all_chunks if not self._is_boilerplate(c)]
        boilerplate = [c for c in all_chunks if self._is_boilerplate(c)]
        if boilerplate:
            logger.debug(
                "MultiStoreRetriever: de-prioritizing %d boilerplate chunks", len(boilerplate)
            )
        return (clean + boilerplate)[: self._top_k]

    def _route(self, query: str, doc_source: str | None) -> list[Any]:
        """Return ordered list of retrievers to query."""
        q = query.lower()
        is_section_query = any(kw in q for kw in _SECTION_QUERY_KEYWORDS)

        if doc_source == "transcript":
            return [self._transcript]

        if doc_source == "sec_filing":
            return [self._sec]

        # No doc_source specified — query both; section queries lead with SEC
        if is_section_query:
            return [self._sec, self._transcript]
        return [self._transcript, self._sec]

    @staticmethod
    def _is_boilerplate(chunk: dict) -> bool:
        text_start = chunk.get("text", "")[:200].lower().strip()
        return any(pat in text_start for pat in _BOILERPLATE_HEADERS)
