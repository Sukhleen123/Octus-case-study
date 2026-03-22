"""
SimFin API v3 realtime client.

References:
  - Getting started (Authorization header): https://simfin.readme.io/reference/getting-started-1
  - Rate limits: https://simfin.readme.io/reference/rate-limits
  - List companies: https://simfin.readme.io/reference/list-1
  - Statements: https://simfin.readme.io/reference/statements-1
  - Major update (base URL): https://www.simfin.com/en/blog/major-simfin-update/
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from src.simfin.cache import RealtimeCache

logger = logging.getLogger(__name__)

# SimFin v3 rate limit: 5 requests/second for free tier (conservative estimate)
_RATE_LIMIT_RPS = 5
_MIN_INTERVAL = 1.0 / _RATE_LIMIT_RPS


class SimFinV3Client:
    """
    HTTP client for SimFin API v3.

    Uses:
    - Authorization header (Bearer token) per v3 spec
    - SQLite cache to avoid redundant API calls
    - Rate limiting via time.sleep between requests

    Args:
        api_key: SimFin API key (set SIMFIN_API_KEY env var).
        base_url: Base URL (default: https://backend.simfin.com, or https://prod.simfin.com).
        cache: RealtimeCache instance.
    """

    def __init__(
        self,
        api_key: str,
        cache: RealtimeCache,
        base_url: str = "https://backend.simfin.com",
    ) -> None:
        self._api_key = api_key
        self._cache = cache
        self._base_url = base_url.rstrip("/")
        self._last_request_time: float = 0.0
        self._client = httpx.Client(
            headers={
                "Authorization": f"api-key {api_key}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """
        Make a GET request, using the cache if available.
        """
        params = params or {}
        cache_key = RealtimeCache.make_key("GET", f"{self._base_url}{path}", params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for %s", path)
            return cached

        self._rate_limit()
        url = f"{self._base_url}{path}"
        logger.debug("GET %s params=%s", url, params)
        response = self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        self._cache.set(cache_key, data)
        return data

    def list_companies(self) -> list[dict[str, Any]]:
        """
        List all companies available in SimFin v3.
        Endpoint: GET /api/v3/companies/list
        """
        return self._get("/api/v3/companies/list")

    def get_statements(
        self,
        ticker: str,
        statement: str = "income",
        period: str = "annual",
        year: int | None = None,
    ) -> Any:
        """
        Retrieve financial statements for a ticker.
        Endpoint: GET /api/v3/companies/statements

        Args:
            ticker: Company ticker symbol.
            statement: "income" | "balance" | "cashflow"
            period: "annual" | "quarterly"
            year: Optional fiscal year filter.
        """
        params: dict[str, Any] = {
            "ticker": ticker,
            "statements": statement,
            "period": period,
        }
        if year:
            params["fyear"] = year
        return self._get("/api/v3/companies/statements", params=params)

    def close(self) -> None:
        self._client.close()
        self._cache.close()

    def __enter__(self) -> "SimFinV3Client":
        return self

    def __exit__(self, *args) -> None:
        self.close()
