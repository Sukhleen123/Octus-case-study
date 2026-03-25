"""
SimFin agent node: retrieves structured financial metrics from DuckDB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.agents import runtime
from src.agents.events import (
    agent_end,
    agent_start,
    citations_emitted,
    simfin_results,
    tool_call_end,
    tool_call_start,
)
from src.agents.state import AgentState
from src.citations.models import SimFinCitation

logger = logging.getLogger(__name__)

NAME = "simfin_agent"

TABLES = (
    "income_annual",   "income_quarterly",
    "balance_annual",  "balance_quarterly",
    "cashflow_annual", "cashflow_quarterly",
)

METRIC_COLS: dict[str, list[str]] = {
    "income": [
        "Revenue",
        "Net Income",
        "Operating Income (Loss)",
        "Gross Profit",
        "Depreciation & Amortization",
    ],
    "balance": ["Total Assets", "Total Equity"],
    "cashflow": ["Net Cash from Operating Activities"],
}


def fetch_ticker(ticker: str) -> tuple[pd.DataFrame, list[SimFinCitation]]:
    """Read financial data for one ticker from DuckDB."""
    duckdb_path = Path(runtime.settings.duckdb_path)

    if not duckdb_path.exists():
        logger.info("SimFinAgent: DuckDB not found at %s", duckdb_path)
        return pd.DataFrame(), []

    try:
        import duckdb

        con = duckdb.connect(str(duckdb_path), read_only=True)
        dfs: list[pd.DataFrame] = []
        for table_name in TABLES:
            try:
                df = con.execute(
                    f'SELECT * FROM {table_name} WHERE upper("Ticker") = upper(?)',
                    [ticker],
                ).df()
                if not df.empty:
                    df["_statement_type"] = table_name
                    dfs.append(df)
            except Exception:
                pass
        con.close()

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(
                "SimFinAgent: loaded %d rows for %s from DuckDB", len(combined), ticker
            )
            return combined, build_citations(combined, ticker)
    except Exception as e:
        logger.warning("SimFinAgent: DuckDB query failed: %s", e)

    logger.info("SimFinAgent: no data found for ticker %s", ticker)
    return pd.DataFrame(), []


def build_citations(df: pd.DataFrame, ticker: str) -> list[SimFinCitation]:
    """Build SimFinCitation objects for the 4 most recent periods."""
    citations: list[SimFinCitation] = []

    year_col = next((c for c in df.columns if "fiscal year" in c.lower()), None)
    period_col = next((c for c in df.columns if "fiscal period" in c.lower()), None)
    stmt_col = "_statement_type"

    sort_col = next(
        (c for c in df.columns if "report date" in c.lower()), period_col
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
        stmt = raw_stmt.split("_")[0]
        metrics = METRIC_COLS.get(stmt, [])

        for metric_name in metrics:
            col = next(
                (c for c in df.columns if metric_name.lower() in c.lower()), None
            )
            if col is None:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue
            unit = "USD per share" if "per share" in metric_name.lower() else "USD thousands"
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


def simfin_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: fetches financial metrics from DuckDB for resolved tickers.

    Sets: simfin_result, trace_events.
    """
    events = []
    events.append(agent_start(NAME, query=state.query, tickers=state.tickers))
    events.append(tool_call_start(NAME, tool="simfin", tickers=state.tickers))

    tables: list[pd.DataFrame] = []
    citations: list[SimFinCitation] = []

    if not state.tickers:
        logger.info("SimFinAgent: no tickers to look up")
        events.append(tool_call_end(NAME, tool="simfin", status="no_tickers"))
        events.append(agent_end(NAME, status="no_tickers"))
        return {
            "simfin_result": (tables, citations, events),
            "trace_events": [e.to_dict() for e in events],
        }

    for ticker in state.tickers:
        try:
            df, cites = fetch_ticker(ticker)
            if not df.empty:
                tables.append(df)
                citations.extend(cites)
        except Exception as e:
            logger.warning("SimFinAgent: failed to fetch %s: %s", ticker, e)

    events.append(tool_call_end(NAME, tool="simfin", tables=len(tables)))
    events.append(simfin_results(NAME, count=len(citations), rows=[c.to_dict() for c in citations]))
    events.append(citations_emitted(NAME, count=len(citations)))
    events.append(agent_end(NAME, table_count=len(tables)))

    return {
        "simfin_result": (tables, citations, events),
        "trace_events": [e.to_dict() for e in events],
    }
