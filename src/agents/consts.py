"""
Shared constants for the agent pipeline.
Consolidated from orchestrator.py and multi_store.py.
"""

from __future__ import annotations

import re

# ── Router / HyDE ─────────────────────────────────────────────────────────────

SIMFIN_KEYWORDS = {
    "revenue", "income", "profit", "loss", "ebitda", "eps", "earnings",
    "cash flow", "balance sheet", "equity", "debt", "assets", "liabilities",
    "margin", "ratio", "fiscal", "quarter", "annual", "10-k", "10-q",
    "financial statement", "net income", "gross profit", "operating",
    "sec filing", "annual report", "quarterly report", "form 10",
}

OCTUS_KEYWORDS = {
    "transcript", "conference", "call", "filing", "sec", "management",
    "said", "stated", "guidance", "outlook", "commentary", "discussion",
    "presentation", "investor", "analyst", "ceo", "cfo",
    "10-k", "10-q", "annual report", "quarterly report",
}

HYDE_TRIGGERS = re.compile(
    r"forward.?looking"
    r"|management\s+(commentary|outlook|guidance|discussion)"
    r"|what\s+did\s+.+\s+say"
    r"|what\s+.+\s+(plan|expect|anticipat|project|forecast)"
    r"|(outlook|guidance|expect|anticipat|project|forecast)\s+for",
    re.IGNORECASE,
)

DOC_TYPE_KEYWORDS: dict[str, str] = {
    "10-k": "10-K",
    "annual report": "10-K",
    "10-q": "10-Q",
    "quarterly report": "10-Q",
}

SEC_KEYWORDS = ("sec filing", "sec report", "form 10")

# ── MultiStoreRetriever ───────────────────────────────────────────────────────

SECTION_QUERY_KEYWORDS = frozenset({
    "risk factor", "item 1a", "md&a", "management discussion",
    "balance sheet", "income statement", "cash flow",
    "notes to financial", "financial statements", "item 7", "item 2",
})

BOILERPLATE_HEADERS = (
    "cautionary note on forward-looking statements",
    "cautionary note regarding forward-looking statements",
    "safe harbor statement under the private securities litigation reform act",
    "note regarding forward-looking statements",
)
