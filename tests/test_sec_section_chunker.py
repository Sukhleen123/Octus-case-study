"""Tests for the SEC section chunker."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.chunking.base import ChunkRecord
from src.chunking.sec_section_chunker import (
    SECSectionChunker,
    _split_sec_sections,
    _table_to_markdown,
)

_REQUIRED_FIELDS = {
    "chunk_id",
    "document_id",
    "doc_source",
    "document_type",
    "document_date",
    "octus_company_id",
    "company_name",
    "company_ids",
    "section_title",
    "chunk_index",
    "char_start",
    "char_end",
    "chunker_id",
    "text",
}


def _make_sec_doc(text: str, doc_source: str = "sec_filing") -> dict:
    """Create a minimal document dict for testing."""
    return {
        "document_id": "sec_doc_001",
        "doc_source": doc_source,
        "document_type": "10-K",
        "document_date": datetime(2024, 3, 15),
        "octus_company_id": "OC_SEC_001",
        "company_name": "Acme Corp",
        "company_ids": "[99999]",
        "cleaned_text": text,
        "raw_path": "",  # no raw file; use cleaned_text
    }


SEC_TEXT = """\
Table of Contents

This is the preamble of the document with a table of contents.

PART I

This is the introductory text for Part I of the filing.

ITEM 1. Financial Statements

The consolidated financial statements include revenue of $500M.
Net income was $50M for the fiscal year ended December 31, 2024.

ITEM 1A. Risk Factors

The company faces several risks including market volatility,
supply chain disruptions, and regulatory changes that could
materially affect operations.

PART II

This section covers additional information.

ITEM 7. MD&A - Management Discussion and Analysis

Management believes the company is well-positioned for growth.
Operating margins improved to 12.5% from 10.2% in the prior year.

ITEM 8. Financial Statements and Supplementary Data

See the accompanying notes to consolidated financial statements.
"""


class TestSplitSecSections:
    """Tests for _split_sec_sections boundary detection."""

    def test_detects_part_boundaries(self):
        sections = _split_sec_sections(SEC_TEXT)
        titles = [t for t, _ in sections]
        part_titles = [t for t in titles if "PART" in t.upper()]
        assert len(part_titles) >= 2, f"Expected >=2 PART sections, got {part_titles}"

    def test_detects_item_boundaries(self):
        sections = _split_sec_sections(SEC_TEXT)
        titles = [t for t, _ in sections]
        item_titles = [t for t in titles if "ITEM" in t.upper() or "Item" in t]
        assert len(item_titles) >= 4, f"Expected >=4 ITEM sections, got {item_titles}"

    def test_preamble_captured(self):
        sections = _split_sec_sections(SEC_TEXT)
        # First section should be the preamble (empty title)
        assert sections[0][0] == ""
        assert "Table of Contents" in sections[0][1] or "preamble" in sections[0][1]

    def test_section_bodies_nonempty(self):
        sections = _split_sec_sections(SEC_TEXT)
        for title, body in sections:
            assert len(body.strip()) > 0, f"Section '{title}' has empty body"

    def test_no_sections_returns_single_block(self):
        plain_text = "This is just plain text with no SEC structure at all."
        sections = _split_sec_sections(plain_text)
        assert len(sections) == 1
        assert sections[0][0] == ""
        assert "plain text" in sections[0][1]

    def test_item_with_period(self):
        text = "Preamble.\n\nITEM 1. Financial Statements\n\nSome content here."
        sections = _split_sec_sections(text)
        titles = [t for t, _ in sections]
        assert any("ITEM 1" in t for t in titles)

    def test_item_with_letter_suffix(self):
        text = "Preamble.\n\nITEM 1A. Risk Factors\n\nRisk content here."
        sections = _split_sec_sections(text)
        titles = [t for t, _ in sections]
        assert any("1A" in t for t in titles)


class TestTableToMarkdown:
    """Tests for _table_to_markdown HTML-to-markdown conversion."""

    def test_basic_table(self):
        from bs4 import BeautifulSoup

        html = """
        <table>
            <tr><th>Quarter</th><th>Revenue</th></tr>
            <tr><td>Q1</td><td>$100M</td></tr>
            <tr><td>Q2</td><td>$120M</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table_tag = soup.find("table")
        md = _table_to_markdown(table_tag)

        assert "Quarter" in md
        assert "Revenue" in md
        assert "$100M" in md
        assert "$120M" in md
        # Should have pipe separators
        assert "|" in md
        # Should have header separator row
        assert "---" in md

    def test_empty_table(self):
        from bs4 import BeautifulSoup

        html = "<table></table>"
        soup = BeautifulSoup(html, "html.parser")
        table_tag = soup.find("table")
        md = _table_to_markdown(table_tag)
        assert md == ""

    def test_pipes_in_cell_text_escaped(self):
        from bs4 import BeautifulSoup

        html = """
        <table>
            <tr><td>A|B</td><td>C</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table_tag = soup.find("table")
        md = _table_to_markdown(table_tag)
        # Pipe in cell text should be replaced with /
        assert "A/B" in md

    def test_uneven_columns_padded(self):
        from bs4 import BeautifulSoup

        html = """
        <table>
            <tr><td>A</td><td>B</td><td>C</td></tr>
            <tr><td>X</td><td>Y</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table_tag = soup.find("table")
        md = _table_to_markdown(table_tag)
        lines = md.strip().splitlines()
        # All lines should have the same number of pipes (columns padded)
        pipe_counts = [line.count("|") for line in lines]
        assert len(set(pipe_counts)) == 1, f"Uneven pipe counts: {pipe_counts}"


class TestSECSectionChunkerTranscriptFallback:
    """Test that transcript documents fall back to heading chunker."""

    def test_transcript_uses_heading_chunker(self):
        chunker = SECSectionChunker()
        doc = _make_sec_doc(
            "## Opening Remarks\nWelcome everyone.\n\n## Q&A\nQuestion one?",
            doc_source="transcript",
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        # Heading chunker produces chunker_id="heading"
        assert all(c.chunker_id == "heading" for c in chunks)


class TestSECSectionChunkerMaxTokens:
    """Test sub-splitting of large sections."""

    def test_large_section_subsplit(self):
        # Create a document with a very large section (> max_section_tokens)
        large_section = "This is a test sentence about financial performance. " * 500
        text = f"ITEM 1. Financial Statements\n\n{large_section}"
        doc = _make_sec_doc(text)
        chunker = SECSectionChunker(max_section_tokens=128, sub_chunk_overlap=16)
        chunks = chunker.chunk(doc)
        # Large section should be split into multiple chunks
        assert len(chunks) > 1

    def test_subsplit_titles_have_part_suffix(self):
        large_section = "This is a test sentence about financial performance. " * 500
        text = f"ITEM 1. Financial Statements\n\n{large_section}"
        doc = _make_sec_doc(text)
        chunker = SECSectionChunker(max_section_tokens=128, sub_chunk_overlap=16)
        chunks = chunker.chunk(doc)
        # Sub-split chunks should have " (part N)" suffix in title
        titled = [c for c in chunks if c.section_title and "(part" in c.section_title]
        assert len(titled) > 0, "Expected sub-split chunks with (part N) title suffix"


class TestSECSectionChunkerOutput:
    """Test chunker_id, field completeness, and output correctness."""

    def setup_method(self):
        self.chunker = SECSectionChunker(max_section_tokens=2048)

    def test_chunker_id_is_sec_section(self):
        doc = _make_sec_doc(SEC_TEXT)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 0
        assert all(c.chunker_id == "sec_section" for c in chunks)

    def test_all_required_fields_populated(self):
        doc = _make_sec_doc(SEC_TEXT)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            d = chunk.to_dict()
            missing = _REQUIRED_FIELDS - set(d.keys())
            assert not missing, f"Missing fields: {missing}"
            # All fields should have non-None values
            for field_name in _REQUIRED_FIELDS:
                assert d[field_name] is not None, f"Field '{field_name}' is None"

    def test_chunk_ids_unique(self):
        doc = _make_sec_doc(SEC_TEXT)
        chunks = self.chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_document_id_propagated(self):
        doc = _make_sec_doc(SEC_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.document_id == "sec_doc_001" for c in chunks)

    def test_chunk_index_sequential(self):
        doc = _make_sec_doc(SEC_TEXT)
        chunks = self.chunker.chunk(doc)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_empty_text_returns_no_chunks(self):
        doc = _make_sec_doc("")
        chunks = self.chunker.chunk(doc)
        assert chunks == []

    def test_text_nonempty_in_chunks(self):
        doc = _make_sec_doc(SEC_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(len(c.text.strip()) > 0 for c in chunks)

    def test_char_positions_reasonable(self):
        doc = _make_sec_doc(SEC_TEXT)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end >= chunk.char_start
            assert chunk.char_end <= len(SEC_TEXT) + 100  # allow small tolerance
