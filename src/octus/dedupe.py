"""
Deduplication of SEC filing metadata records.

Deduplicate key: (tuple(sorted(company_id)), document_type, document_date)

For each group of duplicates, keep the first occurrence as canonical and
record all others as duplicates in the ingest report.
"""

from __future__ import annotations

from typing import Any


def _dedupe_key(record: dict[str, Any]) -> tuple:
    """Build the deduplication key for a SEC filing record."""
    company_ids = record.get("company_id", [])
    # Normalize to sorted tuple for consistent keying
    ids_key = tuple(sorted(int(c) for c in company_ids))
    return (ids_key, record.get("document_type", ""), record.get("document_date", ""))


def dedupe_sec_filings(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Deduplicate SEC filing records by (company_id list, document_type, document_date).

    Args:
        records: Raw list of SEC filing dicts (from JSON).

    Returns:
        (canonical, duplicates) where:
            canonical  - one record per unique key (first occurrence wins)
            duplicates - all non-canonical records, each tagged with their
                         canonical document_id in the "_canonical_id" field
    """
    seen: dict[tuple, dict[str, Any]] = {}
    canonical: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []

    for record in records:
        key = _dedupe_key(record)
        if key not in seen:
            seen[key] = record
            canonical.append(record)
        else:
            dup = dict(record)
            dup["_canonical_id"] = seen[key]["document_id"]
            dup["_dedupe_reason"] = "duplicate_key"
            duplicates.append(dup)

    return canonical, duplicates
