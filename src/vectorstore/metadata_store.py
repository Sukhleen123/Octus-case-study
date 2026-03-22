"""
SQLite-backed metadata store for FAISS vectors.

FAISS is a pure vector similarity library — it has no built-in metadata
filtering (see docs/decisions.md). This store maps vector IDs (int64) to
chunk metadata dicts and supports Python-side filtering after FAISS retrieval.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class MetadataStore:
    """
    Maps FAISS integer IDs to chunk metadata.

    Schema:
        id INTEGER PRIMARY KEY   — FAISS vector index (0-based sequential)
        chunk_id TEXT            — UUID chunk identifier
        metadata TEXT            — JSON blob of all chunk fields
    """

    def __init__(self, db_path: str | Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_meta (
                id INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunk_meta(chunk_id)"
        )
        self._conn.commit()

    def add(self, vector_id: int, chunk_id: str, metadata: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO chunk_meta (id, chunk_id, metadata) VALUES (?, ?, ?)",
            (vector_id, chunk_id, json.dumps(metadata, default=str)),
        )
        self._conn.commit()

    def add_batch(self, records: list[tuple[int, str, dict[str, Any]]]) -> None:
        """Bulk insert: records = [(vector_id, chunk_id, metadata_dict), ...]"""
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunk_meta (id, chunk_id, metadata) VALUES (?, ?, ?)",
            [(vid, cid, json.dumps(meta, default=str)) for vid, cid, meta in records],
        )
        self._conn.commit()

    def get(self, vector_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT metadata FROM chunk_meta WHERE id = ?", (vector_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def get_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Look up metadata by chunk_id string (used by PineconeStore)."""
        row = self._conn.execute(
            "SELECT metadata FROM chunk_meta WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def get_batch_by_chunk_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return {chunk_id: metadata} for a list of chunk_id strings."""
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT chunk_id, metadata FROM chunk_meta WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def add_by_chunk_id(self, chunk_id: str, metadata: dict[str, Any]) -> None:
        """Store metadata keyed only by chunk_id (no integer vector_id needed)."""
        # Use a synthetic integer id based on current row count to satisfy PK constraint
        self._conn.execute(
            """
            INSERT OR REPLACE INTO chunk_meta (id, chunk_id, metadata)
            VALUES (
                COALESCE((SELECT id FROM chunk_meta WHERE chunk_id = ?),
                         (SELECT COALESCE(MAX(id), -1) + 1 FROM chunk_meta)),
                ?, ?)
            """,
            (chunk_id, chunk_id, json.dumps(metadata, default=str)),
        )
        self._conn.commit()

    def add_batch_by_chunk_id(self, records: list[tuple[str, dict[str, Any]]]) -> None:
        """Bulk insert by chunk_id: records = [(chunk_id, metadata_dict), ...]"""
        # Fetch existing chunk_ids to avoid reassigning ids
        existing = {
            row[0]: row[1]
            for row in self._conn.execute("SELECT chunk_id, id FROM chunk_meta").fetchall()
        }
        max_id = self._conn.execute(
            "SELECT COALESCE(MAX(id), -1) FROM chunk_meta"
        ).fetchone()[0]

        rows = []
        for chunk_id, meta in records:
            if chunk_id in existing:
                vid = existing[chunk_id]
            else:
                max_id += 1
                vid = max_id
            rows.append((vid, chunk_id, json.dumps(meta, default=str)))

        self._conn.executemany(
            "INSERT OR REPLACE INTO chunk_meta (id, chunk_id, metadata) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_batch(self, vector_ids: list[int]) -> list[dict[str, Any] | None]:
        placeholders = ",".join("?" * len(vector_ids))
        rows = self._conn.execute(
            f"SELECT id, metadata FROM chunk_meta WHERE id IN ({placeholders})",
            vector_ids,
        ).fetchall()
        id_to_meta = {row[0]: json.loads(row[1]) for row in rows}
        return [id_to_meta.get(vid) for vid in vector_ids]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()[0]

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunk_meta")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
