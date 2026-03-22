"""
FAISS vectorstore with external SQLite metadata store.

Architecture:
  - FAISS IndexFlatIP (inner-product / cosine on unit vectors) stores vectors
  - MetadataStore (SQLite) stores chunk metadata keyed by FAISS int64 ID
  - Metadata filtering is applied Python-side after over-fetching from FAISS

Fact: FAISS is a vector similarity library; it does not support metadata
filtering natively. See: https://github.com/facebookresearch/faiss
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.vectorstore.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

# Over-fetch multiplier for metadata filtering
_OVERFETCH = 10


class FAISSStore:
    """
    Vector store backed by FAISS + SQLite metadata.

    Usage:
        store = FAISSStore(cache_dir, dim)
        store.upsert(vectors, metadatas)
        results = store.query_with_filter(query_vec, k=5, filters={"doc_source": "transcript"})
    """

    def __init__(self, cache_dir: str | Path, dim: int) -> None:
        import faiss  # lazy import so tests can mock it

        self.dim = dim
        self._cache_dir = Path(cache_dir) / "faiss"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._cache_dir / "index.faiss"
        self._meta_db_path = self._cache_dir / "metadata.sqlite"

        self._meta = MetadataStore(self._meta_db_path)

        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            logger.info("Loaded FAISS index (%d vectors)", self._index.ntotal)
        else:
            self._index = faiss.IndexFlatIP(dim)
            logger.info("Created new FAISS index (dim=%d)", dim)

    def upsert(
        self,
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add vectors and their metadata. Vectors must be L2-normalized."""
        import faiss

        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas must have the same length")

        start_id = self._index.ntotal
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        self._index.add(arr)

        records = [
            (start_id + i, meta.get("chunk_id", str(start_id + i)), meta)
            for i, meta in enumerate(metadatas)
        ]
        self._meta.add_batch(records)
        self._save_index()
        logger.info("Upserted %d vectors; index now has %d", len(vectors), self._index.ntotal)

    def query_with_filter(
        self,
        query_vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search FAISS, fetch metadata, apply Python-side filters, return top-k.

        Args:
            query_vector: L2-normalized query embedding.
            k: Number of results to return after filtering.
            filters: Dict of field -> value to match exactly.

        Returns:
            List of metadata dicts for top-k matching chunks.
        """
        if self._index.ntotal == 0:
            return []

        fetch_k = min(k * _OVERFETCH, self._index.ntotal)
        arr = np.array([query_vector], dtype="float32")
        import faiss
        faiss.normalize_L2(arr)
        scores, ids = self._index.search(arr, fetch_k)

        candidate_ids = [int(i) for i in ids[0] if i >= 0]
        metadatas = self._meta.get_batch(candidate_ids)

        results = []
        for vid, meta, score in zip(candidate_ids, metadatas, scores[0]):
            if meta is None:
                continue
            if filters and not _matches(meta, filters):
                continue
            meta = dict(meta)
            meta["_score"] = float(score)
            results.append(meta)
            if len(results) >= k:
                break

        return results

    def _save_index(self) -> None:
        import faiss
        faiss.write_index(self._index, str(self._index_path))

    def count(self) -> int:
        return self._index.ntotal


def _matches(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return True if *meta* satisfies all *filters* (exact match)."""
    for key, value in filters.items():
        if meta.get(key) != value:
            return False
    return True
