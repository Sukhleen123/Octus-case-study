"""
Citation data models.

OctusCitation: references a specific chunk from a transcript or SEC filing.
SimFinCitation: references a specific financial metric from SimFin.
SynthesisResult: the final output from the synthesis agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OctusCitation:
    """Citation pointing to a specific Octus document chunk."""
    document_id: str
    doc_source: str         # "transcript" | "sec_filing"
    document_type: str      # "Transcript" | "10-K" | "10-Q"
    document_date: str      # ISO format datetime string
    chunk_id: str
    cited_text: str = ""    # full text of the retrieved chunk
    company_name: str = ""  # company the chunk belongs to
    ref_number: int = 0     # inline [N] ref assigned by synthesis agent (0 = uncited)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "octus",
            "document_id": self.document_id,
            "doc_source": self.doc_source,
            "document_type": self.document_type,
            "document_date": self.document_date,
            "chunk_id": self.chunk_id,
            "cited_text": self.cited_text,
            "company_name": self.company_name,
            "ref_number": self.ref_number,
        }


@dataclass
class SimFinCitation:
    """Citation pointing to a specific SimFin financial metric."""
    ticker: str
    fiscal_year: int
    fiscal_period: str      # "Q1" | "Q2" | "Q3" | "Q4" | "FY"
    statement_type: str     # "income" | "balance" | "cashflow"
    metric_name: str
    metric_value: str = ""  # actual value as string, e.g. "14588000"
    metric_unit: str = ""   # e.g. "USD thousands" | "USD per share"
    ref_number: int = 0     # inline [N] ref assigned by synthesis agent (0 = uncited)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "simfin",
            "ticker": self.ticker,
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            "statement_type": self.statement_type,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_unit": self.metric_unit,
            "ref_number": self.ref_number,
        }


@dataclass
class SynthesisResult:
    """Final output from the synthesis agent."""
    final_answer_text: str
    citations: list[OctusCitation | SimFinCitation] = field(default_factory=list)
    trace_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer_text": self.final_answer_text,
            "citations": [c.to_dict() for c in self.citations],
            "trace_events": self.trace_events,
        }
