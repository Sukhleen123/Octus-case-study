"""
SimFin agent: retrieves structured financial metrics and returns tables + citations.

Modes:
  batch    — reads local parquet/DuckDB files produced by batch_ingest (simfin_client=None)
  realtime — calls SimFin v3 API via SimFinV3Client
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.events import (
    TraceEvent,
    agent_end,
    agent_start,
    citations_emitted,
    tool_call_end,
    tool_call_start,
)
from src.citations.models import SimFinCitation

logger = logging.getLogger(__name__)


class SimFinAgent:
    """
    Retrieves SimFin financial data for specified tickers.

    In batch mode (simfin_client=None): reads local parquet/DuckDB files.
    In realtime mode: calls SimFin v3 API (with SQLite cache).
    """

    NAME = "simfin_agent"

    def __init__(self, simfin_client: Any, settings: Any) -> None:
        self._client = simfin_client
        self._settings = settings

    def run(
        self,
        query: str,
        tickers: list[str],
    ) -> tuple[list[pd.DataFrame], list[SimFinCitation], list[TraceEvent]]:
        """
        Retrieve financial data for the given tickers.

        Returns:
            (tables, citations, trace_events)
        """
        events: list[TraceEvent] = []
        events.append(agent_start(self.NAME, query=query, tickers=tickers))
        events.append(tool_call_start(self.NAME, tool="simfin", tickers=tickers))

        tables: list[pd.DataFrame] = []
        citations: list[SimFinCitation] = []

        if not tickers:
            logger.info("SimFinAgent: no tickers to look up")
            events.append(tool_call_end(self.NAME, tool="simfin", status="no_tickers"))
            events.append(agent_end(self.NAME, status="no_tickers"))
            return tables, citations, events

        for ticker in tickers:
            try:
                df, cites = self._fetch_ticker(ticker)
                if not df.empty:
                    tables.append(df)
                    citations.extend(cites)
            except Exception as e:
                logger.warning("SimFinAgent: failed to fetch %s: %s", ticker, e)

        events.append(tool_call_end(self.NAME, tool="simfin", tables=len(tables)))
        events.append(citations_emitted(self.NAME, count=len(citations)))
        events.append(agent_end(self.NAME, table_count=len(tables)))

        return tables, citations, events

    def _fetch_ticker(self, ticker: str) -> tuple[pd.DataFrame, list[SimFinCitation]]:
        """Fetch data for one ticker — realtime if client available, else batch."""
        if self._client is not None and hasattr(self._client, "get_statements"):
            return self._fetch_realtime(ticker)
        return self._fetch_batch(ticker)

    def _fetch_realtime(self, ticker: str) -> tuple[pd.DataFrame, list[SimFinCitation]]:
        """Fetch from SimFin v3 API."""
        raw = self._client.get_statements(ticker=ticker)
        # v3 returns a list of statement dicts; wrap in DataFrame best-effort
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            df = pd.DataFrame([raw])
        else:
            df = pd.DataFrame()
        return df, self._build_citations(df, ticker)

    # All table names written by batch_ingest.py
    _BATCH_TABLES = (
        "income_annual",   "income_quarterly",
        "balance_annual",  "balance_quarterly",
        "cashflow_annual", "cashflow_quarterly",
    )

    def _fetch_batch(self, ticker: str) -> tuple[pd.DataFrame, list[SimFinCitation]]:
        """
        Read from locally cached batch files (parquet or DuckDB).

        Collects all available statement types (income, balance, cashflow)
        for both annual and quarterly variants, filters to ticker, and
        returns a combined DataFrame.
        """
        from src.simfin.storage import read_table

        settings = self._settings
        fmt = settings.table_format
        processed_dir = Path(settings.processed_dir) / "simfin"

        all_dfs: list[pd.DataFrame] = []

        # Try parquet/csv files for all statement types
        for table_name in self._BATCH_TABLES:
            p = processed_dir / f"{table_name}.{fmt}"
            if not p.exists():
                continue
            try:
                df = read_table(p)
                ticker_col = next(
                    (c for c in df.columns if c.lower() in ("ticker", "symbol")),
                    None,
                )
                if ticker_col:
                    df = df[df[ticker_col].astype(str).str.upper() == ticker.upper()]
                if not df.empty:
                    df = df.copy()
                    df["_statement_type"] = table_name
                    all_dfs.append(df)
                    logger.debug(
                        "SimFinAgent: %d rows for %s from %s", len(df), ticker, p.name
                    )
            except Exception as e:
                logger.warning("SimFinAgent: could not read %s: %s", p, e)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            logger.info(
                "SimFinAgent: loaded %d rows for %s from batch files", len(combined), ticker
            )
            return combined, self._build_citations(combined, ticker)

        # Try DuckDB as fallback
        duckdb_path = Path(settings.duckdb_path)
        if duckdb_path.exists():
            try:
                import duckdb
                con = duckdb.connect(str(duckdb_path), read_only=True)
                duck_dfs: list[pd.DataFrame] = []
                for table_name in self._BATCH_TABLES:
                    try:
                        df = con.execute(
                            f"SELECT * FROM {table_name} WHERE upper(Ticker) = upper(?)",
                            [ticker],
                        ).df()
                        if not df.empty:
                            df["_statement_type"] = table_name
                            duck_dfs.append(df)
                    except Exception:
                        pass
                con.close()
                if duck_dfs:
                    combined = pd.concat(duck_dfs, ignore_index=True)
                    logger.info(
                        "SimFinAgent: loaded %d rows for %s from DuckDB",
                        len(combined), ticker,
                    )
                    return combined, self._build_citations(combined, ticker)
            except Exception as e:
                logger.warning("SimFinAgent: DuckDB query failed: %s", e)

        logger.info("SimFinAgent: no batch data found for ticker %s", ticker)
        return pd.DataFrame(), []

    # Key metrics to extract per statement type for citations
    _METRIC_COLS: dict[str, list[str]] = {
        "income": [
            "Revenue",
            "Net Income",
            "Earnings Per Share, Diluted",
            "Earnings Per Share, Basic",
            "Operating Income (Loss)",
        ],
        "balance": ["Total Assets", "Total Equity"],
        "cashflow": ["Net Cash from Operating Activities"],
    }

    def _build_citations(self, df: pd.DataFrame, ticker: str) -> list[SimFinCitation]:
        """
        Build one SimFinCitation per (metric, period) for the 4 most recent periods.

        Expands from head(3)/hardcoded Revenue to cover Revenue, Net Income, EPS,
        and other key metrics per statement type, with actual values populated.
        """
        citations: list[SimFinCitation] = []

        year_col = next((c for c in df.columns if "fiscal year" in c.lower()), None)
        period_col = next((c for c in df.columns if "fiscal period" in c.lower()), None)
        stmt_col = "_statement_type"

        # Sort most-recent first using Report Date if available
        sort_col = next(
            (c for c in df.columns if "report date" in c.lower()),
            period_col,
        )
        df_s = df.sort_values(sort_col, ascending=False) if sort_col else df

        seen_periods: list[tuple] = []

        for _, row in df_s.iterrows():
            period_key = (row.get(year_col), row.get(period_col))
            if period_key not in seen_periods:
                seen_periods.append(period_key)
            if len(seen_periods) > 4:
                break

            raw_stmt = str(row.get(stmt_col, "income"))
            stmt = raw_stmt.split("_")[0]  # "income_quarterly" → "income"
            metrics = self._METRIC_COLS.get(stmt, [])

            for metric_name in metrics:
                col = next(
                    (c for c in df.columns if metric_name.lower() in c.lower()),
                    None,
                )
                if col is None:
                    continue
                val = row.get(col)
                if pd.isna(val):
                    continue
                unit = (
                    "USD per share"
                    if "per share" in metric_name.lower()
                    else "USD thousands"
                )
                citations.append(
                    SimFinCitation(
                        ticker=ticker,
                        fiscal_year=int(row[year_col]) if year_col and row.get(year_col) else 0,
                        fiscal_period=str(row[period_col]) if period_col and row.get(period_col) else "FY",
                        statement_type=stmt,
                        metric_name=metric_name,
                        metric_value=str(val),
                        metric_unit=unit,
                    )
                )

            if len(citations) >= 16:
                break

        return citations
