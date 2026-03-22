"""Test that the SimFin realtime cache avoids redundant HTTP calls."""

from __future__ import annotations

import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from src.simfin.cache import RealtimeCache


class TestRealtimeCache:
    def test_cache_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RealtimeCache(cache_dir=tmpdir)
            key = RealtimeCache.make_key("GET", "https://example.com/api", {"ticker": "AAPL"})
            assert cache.get(key) is None

    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RealtimeCache(cache_dir=tmpdir)
            key = RealtimeCache.make_key("GET", "https://example.com/api", {"ticker": "AAPL"})
            data = {"revenue": 1000000, "ticker": "AAPL"}
            cache.set(key, data)
            result = cache.get(key)
            assert result == data

    def test_cache_hit_avoids_http(self):
        """Second call with same params returns cached result, HTTP not called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RealtimeCache(cache_dir=tmpdir)

            mock_http_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": "test_value"}
            mock_response.raise_for_status = MagicMock()
            mock_http_client.get.return_value = mock_response

            with patch("src.simfin.realtime_client_v3.httpx.Client") as mock_client_cls:
                mock_client_cls.return_value.__enter__ = lambda s: mock_http_client
                mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
                mock_client_cls.return_value.get = mock_http_client.get

                from src.simfin.realtime_client_v3 import SimFinV3Client

                client = SimFinV3Client(
                    api_key="test-key",
                    cache=cache,
                    base_url="https://example.com",
                )
                # Inject the mock HTTP client directly
                client._client = mock_http_client

                # First call: cache miss → HTTP
                result1 = client._get("/api/v3/companies/list")
                assert mock_http_client.get.call_count == 1

                # Second call: cache hit → NO HTTP
                result2 = client._get("/api/v3/companies/list")
                assert mock_http_client.get.call_count == 1  # still 1, not 2

                assert result1 == result2

    def test_cache_key_is_deterministic(self):
        """Same inputs always produce same key, regardless of param order."""
        key1 = RealtimeCache.make_key("GET", "https://api.example.com", {"b": 2, "a": 1})
        key2 = RealtimeCache.make_key("GET", "https://api.example.com", {"a": 1, "b": 2})
        assert key1 == key2

    def test_different_params_different_key(self):
        key1 = RealtimeCache.make_key("GET", "https://api.example.com", {"ticker": "AAPL"})
        key2 = RealtimeCache.make_key("GET", "https://api.example.com", {"ticker": "DAL"})
        assert key1 != key2

    def test_expired_entry_returns_none(self):
        """Entries past TTL are treated as missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RealtimeCache(cache_dir=tmpdir, ttl_seconds=1)
            key = RealtimeCache.make_key("GET", "https://example.com", {})
            cache.set(key, {"value": 42})

            # Manually set the created_at to past TTL
            cache._conn.execute(
                "UPDATE cache SET created_at = ? WHERE key = ?",
                (time.time() - 10, key),
            )
            cache._conn.commit()

            result = cache.get(key)
            assert result is None

    def test_clear_expired(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RealtimeCache(cache_dir=tmpdir, ttl_seconds=1)
            key = RealtimeCache.make_key("GET", "https://example.com", {})
            cache.set(key, {"value": 42})

            # Expire it
            cache._conn.execute(
                "UPDATE cache SET created_at = ? WHERE key = ?",
                (time.time() - 10, key),
            )
            cache._conn.commit()

            deleted = cache.clear_expired()
            assert deleted == 1
