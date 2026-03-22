"""
SimFin batch ingestion using the simfin Python package.

Downloads income statements, balance sheets, and cash flow statements
for all mapped Octus companies and writes them to parquet/csv files.

References:
  - SimFin load variants: https://github.com/SimFin/simfin/blob/master/simfin/load.py
  - SimFin Python API docs: https://simfin.readthedocs.io/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.simfin.storage import write_table

logger = logging.getLogger(__name__)

# All statement types downloaded per periodicity variant
_STATEMENT_LOADERS = [
    ("income",   "load_income"),
    ("balance",  "load_balance"),
    ("cashflow", "load_cashflow"),
]


def _get_tickers_from_map(mapping_df: pd.DataFrame, mapping_mode: str) -> list[str]:
    from src.simfin.mapping import filter_by_mapping_mode
    filtered = filter_by_mapping_mode(mapping_df, mapping_mode)
    tickers = filtered["suggested_ticker"].dropna()
    tickers = tickers[tickers != ""].unique().tolist()
    return tickers


def _load_and_write(
    loader_fn: Callable,
    variant: str,
    table_name: str,
    tickers: list[str],
    out_dir: Path,
    fmt: str,
) -> pd.DataFrame | None:
    """Load one statement type from SimFin, filter to Octus tickers, write to disk."""
    try:
        df = loader_fn(variant=variant, market="us")
        if df is None or df.empty:
            logger.info("No data returned for %s (%s)", table_name, variant)
            return None

        # Filter to Octus companies only
        if "Ticker" in df.columns:
            df = df[df["Ticker"].isin(tickers)]
        elif df.index.name == "Ticker":
            df = df[df.index.isin(tickers)]

        if df.empty:
            logger.info("No rows matched Octus tickers for %s", table_name)
            return None

        write_table(df.reset_index(), out_dir / f"{table_name}.{fmt}", fmt)
        logger.info("Wrote %s: %d rows", table_name, len(df))
        return df
    except Exception as e:
        logger.warning("Failed to load %s: %s", table_name, e)
        return None


def run_batch_ingest(settings: Any, company_map: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Download SimFin financial statements via the simfin Python package.

    Downloads income statements, balance sheets, and cash flow for each
    configured periodicity variant, filtered to the Octus company tickers.

    Stores results as parquet/csv files in data/processed/simfin/:
      income_annual.parquet, income_quarterly.parquet
      balance_annual.parquet, balance_quarterly.parquet
      cashflow_annual.parquet, cashflow_quarterly.parquet

    Also writes to DuckDB if duckdb_path is configured.

    Returns:
        Dict of {table_name: DataFrame}
    """
    try:
        import simfin as sf
    except ImportError:
        logger.error("simfin package not installed. Run: pip install simfin")
        return {}

    if not settings.simfin_api_key:
        logger.warning("SIMFIN_API_KEY not set — batch ingestion skipped")
        return {}

    sf.set_api_key(settings.simfin_api_key)
    cache_dir = str(Path(settings.cache_dir) / "simfin_py")
    sf.set_data_dir(cache_dir)

    out_dir = Path(settings.processed_dir) / "simfin"
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = settings.table_format

    tickers = _get_tickers_from_map(company_map, settings.mapping_mode)
    if not tickers:
        logger.warning(
            "No valid tickers in company_map for mapping_mode=%s", settings.mapping_mode
        )
        return {}

    periodicity = settings.simfin_periodicity
    logger.info(
        "Batch ingest for %d tickers, periodicity=%s", len(tickers), periodicity
    )

    results: dict[str, pd.DataFrame] = {}

    for variant, suffix in [("annual", "annual"), ("quarterly", "quarterly")]:
        if suffix == "annual" and periodicity not in ("annual", "both"):
            continue
        if suffix == "quarterly" and periodicity not in ("quarterly", "both"):
            continue

        for stmt_name, loader_attr in _STATEMENT_LOADERS:
            table_name = f"{stmt_name}_{suffix}"
            df = _load_and_write(
                getattr(sf, loader_attr), variant, table_name, tickers, out_dir, fmt
            )
            if df is not None:
                results[table_name] = df

    # Write to DuckDB if configured
    if settings.duckdb_path and results:
        _write_to_duckdb(results, settings.duckdb_path)

    return results


def _write_to_duckdb(dataframes: dict[str, pd.DataFrame], db_path: str) -> None:
    """Write DataFrames as DuckDB tables."""
    try:
        import duckdb
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(db_path)
        for table_name, df in dataframes.items():
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            logger.info("DuckDB table '%s' created (%d rows)", table_name, len(df))
        con.close()
    except Exception as e:
        logger.error("DuckDB write failed: %s", e)
