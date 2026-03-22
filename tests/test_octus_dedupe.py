"""Tests for SEC filing deduplication."""

from __future__ import annotations

from src.octus.dedupe import dedupe_sec_filings


def _make_filing(doc_id: str, company_ids: list[int], doc_type: str, doc_date: str) -> dict:
    return {
        "document_id": doc_id,
        "source_type": "SEC Filing",
        "document_type": doc_type,
        "document_date": doc_date,
        "company_id": company_ids,
    }


class TestDeduplication:
    def test_no_duplicates(self):
        """Unique records are all canonical, no duplicates."""
        records = [
            _make_filing("doc1", [100], "10-K", "20240101 000000"),
            _make_filing("doc2", [200], "10-Q", "20240401 000000"),
        ]
        canonical, duplicates = dedupe_sec_filings(records)
        assert len(canonical) == 2
        assert len(duplicates) == 0

    def test_exact_duplicate(self):
        """Two records with same key → one canonical, one duplicate."""
        records = [
            _make_filing("doc1", [100], "10-K", "20240101 000000"),
            _make_filing("doc2", [100], "10-K", "20240101 000000"),
        ]
        canonical, duplicates = dedupe_sec_filings(records)
        assert len(canonical) == 1
        assert len(duplicates) == 1
        # First occurrence wins
        assert canonical[0]["document_id"] == "doc1"
        # Duplicate points back to canonical
        assert duplicates[0]["_canonical_id"] == "doc1"
        assert duplicates[0]["document_id"] == "doc2"

    def test_company_id_order_doesnt_matter(self):
        """[100, 200] and [200, 100] should produce the same dedupe key."""
        records = [
            _make_filing("doc1", [100, 200], "10-K", "20240101 000000"),
            _make_filing("doc2", [200, 100], "10-K", "20240101 000000"),
        ]
        canonical, duplicates = dedupe_sec_filings(records)
        assert len(canonical) == 1
        assert len(duplicates) == 1

    def test_different_doc_type_not_deduplicated(self):
        """Same company + date but different document_type → both canonical."""
        records = [
            _make_filing("doc1", [100], "10-K", "20240101 000000"),
            _make_filing("doc2", [100], "10-Q", "20240101 000000"),
        ]
        canonical, duplicates = dedupe_sec_filings(records)
        assert len(canonical) == 2
        assert len(duplicates) == 0

    def test_different_date_not_deduplicated(self):
        """Same company + doc_type but different date → both canonical."""
        records = [
            _make_filing("doc1", [100], "10-K", "20240101 000000"),
            _make_filing("doc2", [100], "10-K", "20240401 000000"),
        ]
        canonical, duplicates = dedupe_sec_filings(records)
        assert len(canonical) == 2
        assert len(duplicates) == 0

    def test_three_duplicates(self):
        """Three records with same key → 1 canonical + 2 duplicates."""
        records = [
            _make_filing("doc1", [100], "10-K", "20240101 000000"),
            _make_filing("doc2", [100], "10-K", "20240101 000000"),
            _make_filing("doc3", [100], "10-K", "20240101 000000"),
        ]
        canonical, duplicates = dedupe_sec_filings(records)
        assert len(canonical) == 1
        assert len(duplicates) == 2
        assert all(d["_canonical_id"] == "doc1" for d in duplicates)

    def test_dedupe_reason_field(self):
        """Duplicate records have _dedupe_reason set."""
        records = [
            _make_filing("doc1", [100], "10-K", "20240101 000000"),
            _make_filing("doc2", [100], "10-K", "20240101 000000"),
        ]
        _, duplicates = dedupe_sec_filings(records)
        assert duplicates[0]["_dedupe_reason"] == "duplicate_key"

    def test_empty_input(self):
        canonical, duplicates = dedupe_sec_filings([])
        assert canonical == []
        assert duplicates == []
