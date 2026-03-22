"""
Build metadata filter dicts for Pinecone and FAISS queries.

Pinecone uses server-side filtering with its native filter syntax.
FAISS uses Python-side exact-match filtering after retrieval.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def build_filter(
    company_name: str | None = None,
    doc_source: str | None = None,
    document_type: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> dict[str, Any]:
    """
    Build a simple filter dict suitable for both FAISSStore and PineconeStore.

    FAISSStore accepts {field: value} for exact matching.
    PineconeStore converts this to Pinecone filter syntax internally.

    Date range filters are not supported via exact match — use post-filtering
    for date ranges when using FAISS.
    """
    f: dict[str, Any] = {}
    if company_name:
        f["company_name"] = company_name
    if doc_source:
        f["doc_source"] = doc_source
    if document_type:
        f["document_type"] = document_type
    return f


def apply_date_filter(
    results: list[dict[str, Any]],
    date_from: datetime | None,
    date_to: datetime | None,
) -> list[dict[str, Any]]:
    """
    Post-filter results by document_date range.

    Used after retrieval for both FAISS (where no server-side date filter exists)
    and as an extra guard for Pinecone results.
    """
    if date_from is None and date_to is None:
        return results

    filtered = []
    for r in results:
        doc_date = r.get("document_date")
        if doc_date is None:
            filtered.append(r)
            continue
        # Normalize to datetime
        if isinstance(doc_date, str):
            try:
                doc_date = datetime.fromisoformat(doc_date)
            except ValueError:
                filtered.append(r)
                continue
        if date_from and doc_date < date_from:
            continue
        if date_to and doc_date > date_to:
            continue
        filtered.append(r)

    return filtered
