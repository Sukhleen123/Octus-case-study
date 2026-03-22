"""
SimFin metrics catalog — discover and document available metrics.

Writes data/processed/simfin/metrics_catalog.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Well-known SimFin income statement metrics
INCOME_METRICS = [
    "Revenue", "Cost of Revenue", "Gross Profit", "Operating Income",
    "Net Income", "EPS Diluted", "EBITDA", "Depreciation & Amortization",
    "Interest Expense", "Income Tax Expense",
]

BALANCE_METRICS = [
    "Total Assets", "Total Liabilities", "Total Equity",
    "Cash & Equivalents", "Total Debt", "Net Debt",
]

CASHFLOW_METRICS = [
    "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
    "Free Cash Flow", "Capital Expenditures",
]


def build_metrics_catalog(settings: Any) -> dict[str, Any]:
    """
    Build a metrics catalog from known SimFin schema.

    In a production system, this would query the actual data to discover
    available columns. Here we use a well-known reference set.
    """
    catalog = {
        "income_statement": INCOME_METRICS,
        "balance_sheet": BALANCE_METRICS,
        "cash_flow": CASHFLOW_METRICS,
    }

    out_dir = Path(settings.processed_dir) / "simfin"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_catalog.json"

    with out_path.open("w") as f:
        json.dump(catalog, f, indent=2)

    logger.info("Metrics catalog written to %s", out_path)
    return catalog
