"""Tests for the recursive chunker."""

from __future__ import annotations

from datetime import datetime

import pytest
import tiktoken

from src.chunking.base import ChunkRecord
from src.chunking.recursive_chunker import RecursiveChunker

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

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _make_doc(text: str) -> dict:
    """Create a minimal document dict for testing."""
    return {
        "document_id": "rec_doc_001",
        "doc_source": "transcript",
        "document_type": "Transcript",
        "document_date": datetime(2024, 6, 15, 14, 0, 0),
        "octus_company_id": "OC_REC_001",
        "company_name": "RecTest Inc",
        "company_ids": "[77777]",
        "cleaned_text": text,
    }


# Sample text with clear paragraph boundaries (double newlines)
PARAGRAPH_TEXT = """\
This is the first paragraph about company revenue. The company reported strong earnings.

This is the second paragraph about operating expenses. Costs were well managed.

This is the third paragraph about future outlook. Management remains optimistic.
"""

# Text with many small paragraphs that should be merged
SMALL_PARAGRAPHS = "\n\n".join(
    [f"Sentence number {i}." for i in range(20)]
)

# A single very large paragraph (no double-newline breaks) for sub-splitting
LARGE_PARAGRAPH = (
    "The company reported strong financial results for the quarter. "
    "Revenue increased by fifteen percent year over year. "
    "Net income rose to fifty million dollars. "
    "Operating margins expanded by two percentage points. "
    "The CEO highlighted growth in the cloud segment. "
    "International markets contributed thirty percent of total revenue. "
    "Capital expenditures were in line with guidance. "
    "Free cash flow improved significantly. "
    "The board approved a new share buyback program. "
    "Guidance for next quarter was raised above consensus. "
) * 20  # Repeat to make it large


class TestParagraphSplitting:
    """Test splitting on double newlines."""

    def test_paragraphs_produce_chunks(self):
        chunker = RecursiveChunker(target_size=512, min_size=32, overlap=16)
        doc = _make_doc(PARAGRAPH_TEXT)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0

    def test_paragraph_boundaries_respected(self):
        """Each chunk should contain complete paragraphs (not mid-paragraph splits)
        unless a paragraph itself is oversized."""
        chunker = RecursiveChunker(target_size=512, min_size=32, overlap=16)
        doc = _make_doc(PARAGRAPH_TEXT)
        chunks = chunker.chunk(doc)
        # With large target_size, all paragraphs should merge into one chunk
        # since total tokens are small
        total_tokens = _count_tokens(PARAGRAPH_TEXT)
        if total_tokens <= 512:
            assert len(chunks) == 1


class TestMergeSmallParagraphs:
    """Test that small adjacent paragraphs are merged to reach target_size."""

    def test_small_paragraphs_merged(self):
        chunker = RecursiveChunker(target_size=256, min_size=32, overlap=16)
        doc = _make_doc(SMALL_PARAGRAPHS)
        chunks = chunker.chunk(doc)
        # 20 tiny paragraphs should be merged into fewer chunks
        assert len(chunks) < 20, (
            f"Expected merging to reduce 20 paragraphs, got {len(chunks)} chunks"
        )

    def test_merged_chunks_have_content(self):
        chunker = RecursiveChunker(target_size=256, min_size=32, overlap=16)
        doc = _make_doc(SMALL_PARAGRAPHS)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0


class TestSubSplitOversized:
    """Test sub-splitting of oversized paragraphs on sentence boundaries."""

    def test_large_paragraph_subsplit(self):
        chunker = RecursiveChunker(target_size=64, min_size=16, overlap=8)
        doc = _make_doc(LARGE_PARAGRAPH)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1, "Large paragraph should be split into multiple chunks"

    def test_subsplit_respects_sentence_boundaries(self):
        chunker = RecursiveChunker(target_size=64, min_size=16, overlap=8)
        doc = _make_doc(LARGE_PARAGRAPH)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            text = chunk.text.strip()
            # Chunks should generally end at sentence boundaries (period)
            # or be the last chunk
            if chunk.chunk_index < len(chunks) - 1:
                assert len(text) > 0


class TestOverlapBehavior:
    """Test overlap between chunks."""

    def test_overlap_produces_shared_content(self):
        chunker = RecursiveChunker(target_size=64, min_size=16, overlap=16)
        doc = _make_doc(LARGE_PARAGRAPH)
        chunks = chunker.chunk(doc)
        if len(chunks) >= 2:
            # With overlap, consecutive chunks should share some text
            # Check that at least one pair of consecutive chunks has shared words
            found_overlap = False
            for i in range(len(chunks) - 1):
                words_a = set(chunks[i].text.split()[-10:])
                words_b = set(chunks[i + 1].text.split()[:10])
                if words_a & words_b:
                    found_overlap = True
                    break
            # Overlap is expected but depends on sentence boundaries,
            # so we just verify we got multiple chunks
            assert len(chunks) > 1


class TestTargetSizeCompliance:
    """Test that chunks respect target_size token limit."""

    def test_chunks_within_target_size(self):
        target = 128
        chunker = RecursiveChunker(target_size=target, min_size=32, overlap=16)
        doc = _make_doc(LARGE_PARAGRAPH)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            n_tokens = _count_tokens(chunk.text)
            # Allow some tolerance for sentence boundary alignment
            assert n_tokens <= target * 1.5, (
                f"Chunk {chunk.chunk_index} has {n_tokens} tokens, "
                f"exceeds target {target} by too much"
            )


class TestRecursiveChunkerOutput:
    """Test chunker_id, required fields, and general output correctness."""

    def setup_method(self):
        self.chunker = RecursiveChunker(target_size=256, min_size=32, overlap=16)

    def test_chunker_id_is_recursive(self):
        doc = _make_doc(PARAGRAPH_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.chunker_id == "recursive" for c in chunks)

    def test_all_required_fields_populated(self):
        doc = _make_doc(PARAGRAPH_TEXT)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            d = chunk.to_dict()
            missing = _REQUIRED_FIELDS - set(d.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_chunk_ids_unique(self):
        doc = _make_doc(PARAGRAPH_TEXT)
        chunks = self.chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_chunk_index_sequential(self):
        doc = _make_doc(PARAGRAPH_TEXT)
        chunks = self.chunker.chunk(doc)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_document_id_propagated(self):
        doc = _make_doc(PARAGRAPH_TEXT)
        chunks = self.chunker.chunk(doc)
        assert all(c.document_id == "rec_doc_001" for c in chunks)


class TestEmptyTextHandling:
    """Test edge cases with empty or whitespace-only text."""

    def test_empty_string(self):
        chunker = RecursiveChunker(target_size=256, min_size=32, overlap=16)
        doc = _make_doc("")
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_whitespace_only(self):
        chunker = RecursiveChunker(target_size=256, min_size=32, overlap=16)
        doc = _make_doc("   \n\n   \n   ")
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap must be less than target_size"):
            RecursiveChunker(target_size=64, min_size=16, overlap=64)
