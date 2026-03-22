"""
Date parsing and normalization helpers for Octus data.

Raises ValueError immediately on malformed dates (fail-fast policy).
"""

from __future__ import annotations

from datetime import datetime


def parse_transcript_date(s: str) -> datetime:
    """
    Parse a transcript document_date string.
    Expected format: "YYYY-MM-DD HH:MM:SS"

    Raises:
        ValueError if the string does not match the expected format.
    """
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError(
            f"Cannot parse transcript date {s!r}. Expected format: 'YYYY-MM-DD HH:MM:SS'"
        )


def parse_sec_date(s: str) -> datetime:
    """
    Parse a SEC filing document_date string.
    Expected format: "YYYYMMDD HHMMSS"

    Raises:
        ValueError if the string does not match the expected format.
    """
    try:
        return datetime.strptime(s, "%Y%m%d %H%M%S")
    except ValueError:
        raise ValueError(
            f"Cannot parse SEC date {s!r}. Expected format: 'YYYYMMDD HHMMSS'"
        )


def parse_company_ids(s: str) -> list[int]:
    """
    Parse the company_ids comma-separated string from company_metadata.json.

    Returns an empty list if the string is empty or None.
    """
    if not isinstance(s, str) or not s.strip():
        return []
    result = []
    for part in s.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    return result
