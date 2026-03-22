"""
Pinecone vectorstore.

Pinecone supports metadata filtering at query time via its filter parameter.
See: https://docs.pinecone.io/guides/search/filter-by-metadata

Pinecone limits metadata to 40 KB per vector.  Chunk text (especially from
heading-based chunking of SEC filings) easily exceeds this.  Solution:

  - `text` is stored in a local SQLite MetadataStore keyed by chunk_id
  - Only filterable fields (company_name, doc_source, etc.) go into Pinecone
  - After query, text is joined back from the local store by chunk_id

Metadata values in Pinecone must be flat (str, int, float, bool, or list[str]).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fields that must NOT be sent to Pinecone (exceed 40 KB limit for large docs)
_PINECONE_EXCLUDED_FIELDS = {"text"}

# Fields whose string length should be capped as a safety measure
_MAX_FIELD_LEN = 500
_TRUNCATED_FIELDS = {"section_title"}


class PineconeStore:
    """
    Vectorstore backed by a Pinecone index + local SQLite text store.

    Args:
        api_key: Pinecone API key.
        index_name: Name of the Pinecone index.
        namespace: Pinecone namespace.
        dim: Embedding dimension (used when creating a new index).
        text_store: MetadataStore instance for storing chunk text locally.
                    Required — text cannot be stored in Pinecone due to size limits.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str,
        dim: int,
        text_store: Any,  # MetadataStore
    ) -> None:
        from pinecone import Pinecone, ServerlessSpec

        self._pc = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._namespace = namespace
        self.dim = dim
        self._text_store = text_store

        # Create index if it doesn't exist
        existing = [idx.name for idx in self._pc.list_indexes()]
        if index_name not in existing:
            logger.info("Creating Pinecone index '%s' (dim=%d)", index_name, dim)
            self._pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self._index = self._pc.Index(index_name)
        logger.info(
            "Connected to Pinecone index '%s' namespace='%s'", index_name, namespace
        )

    def upsert(
        self,
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Upsert vectors into Pinecone.

        - Saves full chunk text to local MetadataStore (avoids 40 KB limit)
        - Sends only filterable fields to Pinecone metadata
        """
        # Save text to local store in bulk
        text_records = [
            (meta["chunk_id"], {"text": meta.get("text", ""), "chunk_id": meta["chunk_id"]})
            for meta in metadatas
            if meta.get("chunk_id")
        ]
        if text_records:
            self._text_store.add_batch_by_chunk_id(text_records)

        # Build Pinecone records without text
        records = []
        for vec, meta in zip(vectors, metadatas):
            vector_id = meta.get("chunk_id", "")
            if not vector_id:
                raise ValueError("metadata must contain 'chunk_id'")
            records.append({
                "id": vector_id,
                "values": vec,
                "metadata": _flatten_metadata(meta),
            })

        batch_size = 100
        for i in range(0, len(records), batch_size):
            self._index.upsert(vectors=records[i : i + batch_size], namespace=self._namespace)

        logger.info("Upserted %d vectors to Pinecone", len(records))

    def query_with_filter(
        self,
        query_vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query Pinecone and rejoin chunk text from local store.

        Returns:
            List of metadata dicts with `text` field populated from local store.
        """
        pinecone_filter = _to_pinecone_filter(filters) if filters else None

        response = self._index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            namespace=self._namespace,
            filter=pinecone_filter,
        )

        results = []
        chunk_ids = [match.id for match in response.matches]

        # Batch-fetch texts from local store
        local_data = self._text_store.get_batch_by_chunk_ids(chunk_ids) if chunk_ids else {}

        for match in response.matches:
            meta = dict(match.metadata or {})
            meta["_score"] = match.score
            meta["chunk_id"] = match.id
            # Rejoin text from local store
            local = local_data.get(match.id, {})
            meta["text"] = local.get("text", "")
            results.append(meta)

        return results

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        ns_stats = stats.namespaces.get(self._namespace)
        return ns_stats.vector_count if ns_stats else 0


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Prepare metadata for Pinecone:
    - Exclude large fields (text)
    - Truncate fields that could be long (section_title)
    - Convert datetime to ISO string
    - Ensure all values are flat types
    """
    from datetime import datetime
    flat = {}
    for k, v in meta.items():
        if k in _PINECONE_EXCLUDED_FIELDS:
            continue
        if isinstance(v, datetime):
            flat[k] = v.isoformat()
        elif isinstance(v, (str, int, float, bool)):
            if k in _TRUNCATED_FIELDS and isinstance(v, str) and len(v) > _MAX_FIELD_LEN:
                flat[k] = v[:_MAX_FIELD_LEN]
            else:
                flat[k] = v
        elif isinstance(v, list):
            flat[k] = [str(x) for x in v]
        else:
            flat[k] = str(v)
    return flat


def _to_pinecone_filter(filters: dict[str, Any]) -> dict[str, Any]:
    """Convert simple {field: value} filter to Pinecone filter syntax."""
    return {k: {"$eq": v} for k, v in filters.items()}
