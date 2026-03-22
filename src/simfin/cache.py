"""
SQLite cache for SimFin v3 realtime API responses.

Keyed by sha256(method + url + sorted query params).
Prevents redundant HTTP calls and respects rate limits.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class RealtimeCache:
    """SQLite-backed HTTP response cache for SimFin v3 API."""

    def __init__(self, cache_dir: str | Path, ttl_seconds: int = 3600) -> None:
        db_path = Path(cache_dir) / "simfin_realtime_cache.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._ttl = ttl_seconds
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def make_key(method: str, url: str, params: dict[str, Any]) -> str:
        """Deterministic cache key from request parameters."""
        canonical = f"{method.upper()}:{url}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Return cached response or None if missing/expired."""
        row = self._conn.execute(
            "SELECT response, created_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        response_json, created_at = row
        if time.time() - created_at > self._ttl:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        return json.loads(response_json)

    def set(self, key: str, response: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(response, default=str), time.time()),
        )
        self._conn.commit()

    def clear_expired(self) -> int:
        cutoff = time.time() - self._ttl
        cursor = self._conn.execute(
            "DELETE FROM cache WHERE created_at < ?", (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()
