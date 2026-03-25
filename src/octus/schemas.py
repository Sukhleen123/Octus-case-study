"""
Pydantic models for Octus raw data and the normalized OctusDocument.

Only fields that actually exist in the raw JSON files are modelled here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Raw JSON schemas ───────────────────────────────────────────────────────────

class CompanyMeta(BaseModel):
    """One record from company_metadata.json."""
    octus_company_id: str
    company_name: str
    sub_industry: str
    company_ids: str  # comma-separated string, e.g. "625522, 628431, ..."


class TranscriptRecord(BaseModel):
    """One record from transcripts.json."""
    document_id: str
    source_type: str        # "Transcript"
    document_type: str      # "Transcript"
    document_date: str      # format "YYYY-MM-DD HH:MM:SS"
    company_id: int         # single integer entity ID
    body: str               # HTML content


class SecFilingRecord(BaseModel):
    """One record from sec_filings_metadata.json."""
    document_id: str
    source_type: str        # "SEC Filing"
    document_type: str      # "10-Q" or "10-K"
    document_date: str      # format "YYYYMMDD HHMMSS"
    company_id: list[int]   # list of integer entity IDs (1-2 entries)


# ── Normalized document schema ─────────────────────────────────────────────────

class OctusDocument(BaseModel):
    """
    Unified normalized representation written to documents.{parquet|csv}.

    company_ids is stored as a JSON string for compatibility with Parquet
    (pyarrow does not handle list[int] columns cleanly in all scenarios).
    """
    doc_source: str             # "transcript" | "sec_filing"
    document_id: str
    source_type: str
    document_type: str
    document_date: datetime
    octus_company_id: str
    company_name: str
    company_ids: str            # JSON-encoded list[int], e.g. "[625522, 628431]"
    raw_path: str               # path to source file or "" for transcript (body inline)
    cleaned_text: str
