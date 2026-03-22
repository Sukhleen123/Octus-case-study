"""
Router agent: decide whether a query needs Octus docs, SimFin data, or both.

Uses keyword matching as the default strategy. Replace with an LLM call
once an API key is available by setting a custom router in AppContext.
"""

from __future__ import annotations

from typing import Literal

# Keywords that strongly suggest financial statement / metric queries
_SIMFIN_KEYWORDS = {
    "revenue", "income", "profit", "loss", "ebitda", "eps", "earnings",
    "cash flow", "balance sheet", "equity", "debt", "assets", "liabilities",
    "margin", "ratio", "fiscal", "quarter", "annual", "10-k", "10-q",
    "financial statement", "net income", "gross profit", "operating",
    # SEC document types also contain structured financial data:
    "sec filing", "annual report", "quarterly report", "form 10",
}

# Keywords that suggest unstructured document queries
_OCTUS_KEYWORDS = {
    "transcript", "conference", "call", "filing", "sec", "management",
    "said", "stated", "guidance", "outlook", "commentary", "discussion",
    "presentation", "investor", "analyst", "ceo", "cfo",
    # 10-K / 10-Q filings also contain rich narrative text (MD&A, risk factors, etc.):
    "10-k", "10-q", "annual report", "quarterly report",
}

RouteDecision = Literal["octus", "simfin", "both"]


def route(query: str) -> RouteDecision:
    """
    Determine which data sources to query based on the question text.

    Strategy:
    - Count keyword matches in each category
    - If both categories have matches → "both"
    - If only one has matches → that category
    - If neither → "both" (safe default)
    """
    q = query.lower()
    words = set(q.split())

    simfin_hits = sum(1 for kw in _SIMFIN_KEYWORDS if kw in q)
    octus_hits = sum(1 for kw in _OCTUS_KEYWORDS if kw in q)

    if simfin_hits > 0 and octus_hits > 0:
        return "both"
    if simfin_hits > octus_hits:
        return "simfin"
    if octus_hits > simfin_hits:
        return "octus"
    return "both"  # safe default — search everything
