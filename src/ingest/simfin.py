"""
SimFin ingestion pipeline.

Pulls financial statements (PL, BS, CF) for all mapped companies from the
SimFin v3 WebAPI and writes them to DuckDB.

Tables written (one row per company-period):
  income_annual     — Profit & Loss, Fiscal Period == "FY"
  income_quarterly  — Profit & Loss, all other periods (Q1-Q4, H1, 9M, etc.)
  balance_annual    — Balance Sheet, Fiscal Period == "FY"
  balance_quarterly — Balance Sheet, all other periods
  cashflow_annual   — Cash Flow, Fiscal Period == "FY"
  cashflow_quarterly — Cash Flow, all other periods

Usage:
    from src.ingest.simfin import ingest_simfin
    ingest_simfin(settings)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.settings import Settings

logger = logging.getLogger(__name__)

# Map SimFin statement codes to table name prefixes
_STATEMENT_MAP = {"PL": "income", "BS": "balance", "CF": "cashflow"}

# DuckDB table names written by this module (used by SimFinAgent to query)
SIMFIN_TABLES = (
    "income_annual",
    "income_quarterly",
    "balance_annual",
    "balance_quarterly",
    "cashflow_annual",
    "cashflow_quarterly",
)


def ingest_simfin(settings: Settings) -> None:
    """
    Pull SimFin financial data for all mapped companies and write to DuckDB.

    Steps:
      1. Build company map to get confirmed/auto-matched tickers
      2. For each ticker: fetch PL + BS + CF via the v3 verbose endpoint
      3. Normalize: split by statement type and period (annual vs. sub-annual)
      4. Write all 6 tables to DuckDB (full replace on each run)
    """
    from src.simfin.cache import RealtimeCache
    from src.simfin.mapping import build_company_map, filter_by_mapping_mode
    from src.simfin.realtime_client_v3 import SimFinV3Client

    if not settings.simfin_api_key:
        logger.error("SIMFIN_API_KEY is not set — cannot ingest SimFin data")
        return

    logger.info("Building company map...")
    company_map = build_company_map(settings)
    filtered = filter_by_mapping_mode(company_map, settings.mapping_mode)
    tickers = (
        filtered["suggested_ticker"]
        .dropna()
        .loc[lambda s: s.str.strip() != ""]
        .unique()
        .tolist()
    )
    logger.info("Fetching SimFin data for %d tickers: %s", len(tickers), tickers)

    cache = RealtimeCache(cache_dir=settings.cache_dir)
    client = SimFinV3Client(
        api_key=settings.simfin_api_key,
        cache=cache,
        base_url=settings.simfin_base_url,
    )

    # Accumulate DataFrames per table across all tickers
    accumulated: dict[str, list[pd.DataFrame]] = {t: [] for t in SIMFIN_TABLES}

    with client:
        for ticker in tickers:
            try:
                logger.info("Fetching statements for %s...", ticker)
                response = client.get_verbose_statements(ticker, statements="PL,BS,CF")
                frames = _normalize_response(response)
                for table_name, df in frames.items():
                    if not df.empty:
                        accumulated[table_name].append(df)
                        logger.debug(
                            "%s → %s: %d rows", ticker, table_name, len(df)
                        )
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", ticker, e)

    _write_to_duckdb(accumulated, settings)


def _normalize_response(response: Any) -> dict[str, pd.DataFrame]:
    """
    Parse the verbose API response into 6 DataFrames (statement × period type).

    The v3 verbose endpoint returns a list containing one company object:
      [{
        "ticker": "DAL",
        "name": "DELTA AIR LINES ...",
        "id": 231789,
        "statements": [
          {"statement": "PL", "data": [{...}, ...]},
          {"statement": "BS", "data": [{...}, ...]},
          {"statement": "CF", "data": [{...}, ...]}
        ]
      }]

    Annual rows:    Fiscal Period == "FY"
    Quarterly rows: Fiscal Period in Q1/Q2/Q3/Q4/H1/H2/9M/etc.
    """
    result: dict[str, pd.DataFrame] = {}

    # The v3 API wraps the response in a list — unwrap the first element
    if isinstance(response, list):
        if not response:
            return result
        response = response[0]

    if not isinstance(response, dict):
        logger.warning("Unexpected response type: %s", type(response))
        return result

    ticker = response.get("ticker", "")
    company_name = response.get("name", "")
    company_id = response.get("id")

    for stmt_block in response.get("statements", []):
        stmt_code = stmt_block.get("statement", "")
        table_prefix = _STATEMENT_MAP.get(stmt_code)
        if table_prefix is None:
            continue

        data = stmt_block.get("data", [])
        if not data:
            continue

        # Attach company identifiers to every row
        records = []
        for row in data:
            record = dict(row)
            record["Ticker"] = ticker
            record["Company Name"] = company_name
            record["SimFin ID"] = company_id
            records.append(record)

        df = pd.DataFrame(records)

        # Balance sheets use Q4 (year-end snapshot) instead of FY for annual data.
        # Income statements and cash flow use FY for annual totals.
        if stmt_code == "BS":
            annual_mask = df["Fiscal Period"] == "Q4"
            quarterly_mask = df["Fiscal Period"].isin(["Q1", "Q2", "Q3"])
        else:
            annual_mask = df["Fiscal Period"] == "FY"
            quarterly_mask = ~annual_mask

        annual_df = df[annual_mask].copy()
        quarterly_df = df[quarterly_mask].copy()

        if not annual_df.empty:
            result[f"{table_prefix}_annual"] = annual_df
        if not quarterly_df.empty:
            result[f"{table_prefix}_quarterly"] = quarterly_df

    return result


def _write_to_duckdb(
    accumulated: dict[str, list[pd.DataFrame]],
    settings: Settings,
) -> None:
    """Write all accumulated DataFrames to DuckDB, replacing existing tables."""
    import duckdb

    duckdb_path = Path(settings.duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(duckdb_path))
    try:
        for table_name, frames in accumulated.items():
            if not frames:
                logger.info("No data for table '%s' — skipping", table_name)
                continue

            combined = pd.concat(frames, ignore_index=True)
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM combined")
            logger.info(
                "Wrote %d rows to DuckDB table '%s'", len(combined), table_name
            )
    finally:
        con.close()
