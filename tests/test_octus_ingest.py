"""Tests for Octus date parsing and SEC HTML verification."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.octus.normalize import parse_sec_date, parse_transcript_date


# ── Date parsing ───────────────────────────────────────────────────────────────

class TestTranscriptDateParsing:
    def test_valid_date(self):
        result = parse_transcript_date("2024-09-12 21:32:11")
        assert result == datetime(2024, 9, 12, 21, 32, 11)

    def test_valid_date_zero_time(self):
        result = parse_transcript_date("2024-01-01 00:00:00")
        assert result == datetime(2024, 1, 1, 0, 0, 0)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse transcript date"):
            parse_transcript_date("20240912 213211")  # SEC format, not transcript

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_transcript_date("")

    def test_completely_wrong_format(self):
        with pytest.raises(ValueError):
            parse_transcript_date("not-a-date")


class TestSecDateParsing:
    def test_valid_date(self):
        result = parse_sec_date("20241010 161649")
        assert result == datetime(2024, 10, 10, 16, 16, 49)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse SEC date"):
            parse_sec_date("2024-10-10 16:16:49")  # transcript format, not SEC

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_sec_date("")


# ── SEC HTML presence verification ────────────────────────────────────────────

class TestSecHtmlVerification:
    def test_missing_html_raises_when_required(self):
        """run_ingestion raises FileNotFoundError if HTML is missing and require_sec_html=True."""
        from src.octus.ingest import ingest_sec_filings

        fake_filing = {
            "document_id": "nonexistent_doc",
            "source_type": "SEC Filing",
            "document_type": "10-K",
            "document_date": "20240101 000000",
            "company_id": [12345],
        }
        entity_map = {12345: ("OC001", "Test Company")}

        with tempfile.TemporaryDirectory() as tmpdir:
            sec_html_dir = Path(tmpdir) / "sec_html"
            sec_html_dir.mkdir()

            with pytest.raises(FileNotFoundError):
                ingest_sec_filings(
                    [fake_filing], entity_map, sec_html_dir, require_html=True
                )

    def test_missing_html_warns_when_not_required(self):
        """run_ingestion logs warning (not raises) if require_sec_html=False."""
        import logging
        from src.octus.ingest import ingest_sec_filings

        fake_filing = {
            "document_id": "nonexistent_doc",
            "source_type": "SEC Filing",
            "document_type": "10-K",
            "document_date": "20240101 000000",
            "company_id": [12345],
        }
        entity_map = {12345: ("OC001", "Test Company")}

        with tempfile.TemporaryDirectory() as tmpdir:
            sec_html_dir = Path(tmpdir) / "sec_html"
            sec_html_dir.mkdir()

            # Should NOT raise
            docs, dups = ingest_sec_filings(
                [fake_filing], entity_map, sec_html_dir, require_html=False
            )
            assert len(docs) == 1
            assert docs[0]["cleaned_text"] == ""

    def test_existing_html_is_read(self):
        """When HTML file exists, cleaned_text is populated."""
        from src.octus.ingest import ingest_sec_filings

        with tempfile.TemporaryDirectory() as tmpdir:
            sec_html_dir = Path(tmpdir) / "sec_html"
            sec_html_dir.mkdir()

            doc_id = "test_filing_001"
            html_content = "<html><body><h1>Test Filing</h1><p>Revenue: $1B</p></body></html>"
            (sec_html_dir / f"{doc_id}.html").write_text(html_content, encoding="utf-8")

            filing = {
                "document_id": doc_id,
                "source_type": "SEC Filing",
                "document_type": "10-K",
                "document_date": "20240101 000000",
                "company_id": [12345],
            }
            entity_map = {12345: ("OC001", "Test Company")}

            docs, dups = ingest_sec_filings(
                [filing], entity_map, sec_html_dir, require_html=True
            )
            assert len(docs) == 1
            assert "Test Filing" in docs[0]["cleaned_text"]
            assert "Revenue" in docs[0]["cleaned_text"]
