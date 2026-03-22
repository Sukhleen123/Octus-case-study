"""
Disk-based embedding cache (SQLite).

Avoids re-embedding the same text on repeated bootstrap runs.
Keyed by (provider_model, text_sha256).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class EmbeddingCache:
    """SQLite-backed cache for embedding vectors."""

    def __init__(self, cache_dir: str | Path, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model
        db_path = Path(cache_dir) / "embeddings" / "embedding_cache.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                vector TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{self._provider}:{self._model}:{text_hash}"

    def get(self, text: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE key = ?", (self._key(text),)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, text: str, vector: list[float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            (self._key(text), json.dumps(vector)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
