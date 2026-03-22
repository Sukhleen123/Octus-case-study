"""Tests for chunker output schema correctness."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.chunking.heading_chunker import HeadingChunker
from src.chunking.token_chunker import TokenChunker

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


def _make_doc(text: str) -> dict:
    return {
        "document_id": "test_doc_001",
        "doc_source": "transcript",
        "document_type": "Transcript",
        "document_date": datetime(2024, 9, 12, 21, 32, 11),
        "octus_company_id": "OC001",
        "company_name": "Test Co",
        "company_ids": "[12345]",
        "cleaned_text": text,
    }


SAMPLE_TEXT = """
## Introduction
This is the first section of the document. It contains some introductory remarks.

## Financial Results
Revenue increased by 15% year-over-year. Operating margin improved significantly.

## Outlook
Management expects continued growth in Q4 2024.
"""

LONG_TEXT = "This is a sentence about financial performance. " * 200


class TestHeadingChunker:
    def setup_method(self):
        self.chunker = HeadingChunker()

    def test_produces_chunks(self):
        doc = _make_doc(SAMPLE_TEXT)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 0

    def test_chunk_schema(self):
        doc = _make_doc(SAMPLE_TEXT)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            d = chunk.to_dict()
            missing = _REQUIRED_FIELDS - set(d.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_chunk_ids_unique(self):
        doc = _make_doc(SAMPLE_TEXT)
        chunks = self.chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_chunker_id_correct(self):
        doc = _make_doc(SAMPLE_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.chunker_id == "heading" for c in chunks)

    def test_document_id_propagated(self):
        doc = _make_doc(SAMPLE_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.document_id == "test_doc_001" for c in chunks)

    def test_chunk_index_sequential(self):
        doc = _make_doc(SAMPLE_TEXT)
        chunks = self.chunker.chunk(doc)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_empty_text_returns_no_chunks(self):
        doc = _make_doc("")
        chunks = self.chunker.chunk(doc)
        assert chunks == []

    def test_text_nonempty_in_chunks(self):
        doc = _make_doc(SAMPLE_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(len(c.text.strip()) > 0 for c in chunks)


class TestTokenChunker:
    def setup_method(self):
        self.chunker = TokenChunker(chunk_size=50, chunk_overlap=10)

    def test_produces_chunks(self):
        doc = _make_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunk_schema(self):
        doc = _make_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            d = chunk.to_dict()
            missing = _REQUIRED_FIELDS - set(d.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_chunk_ids_unique(self):
        doc = _make_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_chunker_id_correct(self):
        doc = _make_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.chunker_id == "token" for c in chunks)

    def test_chunk_size_respected(self):
        """Each chunk should contain <= chunk_size tokens (approx)."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        doc = _make_doc(LONG_TEXT)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            n_tokens = len(enc.encode(chunk.text))
            assert n_tokens <= 55, f"Chunk too long: {n_tokens} tokens"

    def test_empty_text_returns_no_chunks(self):
        doc = _make_doc("")
        chunks = self.chunker.chunk(doc)
        assert chunks == []

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TokenChunker(chunk_size=50, chunk_overlap=50)
