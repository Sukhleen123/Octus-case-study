"""
Recursive/semantic chunker.

Splits text on paragraph boundaries first, then merges small paragraphs
or sub-splits large ones to reach a target token size. Preserves section
headings as metadata when detected.

Rationale:
  Paragraph-aware splitting produces more semantically coherent chunks
  than a blind token window. Merging small paragraphs avoids fragments
  that lack context; sub-splitting large paragraphs on sentence
  boundaries ensures the result stays within embedding token limits.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import tiktoken

from src.chunking.base import BaseChunker, ChunkRecord
from src.chunking.heading_chunker import _HEADING_RE

_DEFAULT_ENCODING = "cl100k_base"

# Sentence boundary regex: split on ". ", "? ", "! " followed by uppercase
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class RecursiveChunker(BaseChunker):
    """
    Paragraph-aware chunker with merge/split to target size.

    Strategy:
      1. Split on double newlines (paragraph boundaries)
      2. Merge adjacent small paragraphs until target_size tokens
      3. Sub-split oversized paragraphs on sentence boundaries with overlap

    Args:
        target_size: Target tokens per chunk.
        min_size: Minimum tokens before a chunk is merged with the next.
        overlap: Token overlap when sub-splitting oversized paragraphs.
    """

    chunker_id = "recursive"

    def __init__(
        self,
        target_size: int = 512,
        min_size: int = 128,
        overlap: int = 64,
    ) -> None:
        if overlap >= target_size:
            raise ValueError("overlap must be less than target_size")
        self.target_size = target_size
        self.min_size = min_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding(_DEFAULT_ENCODING)

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _detect_heading(self, text: str) -> str:
        """Return the first heading found in text, or empty string."""
        m = _HEADING_RE.search(text)
        if m:
            return (m.group("md_title") or m.group("html_title") or "").strip()
        return ""

    def _split_sentences(self, text: str) -> list[str]:
        """Split text on sentence boundaries."""
        parts = _SENTENCE_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _sub_split(self, text: str) -> list[str]:
        """Split an oversized paragraph into target_size chunks with overlap."""
        sentences = self._split_sentences(text)
        if not sentences:
            # Fall back to token-window
            tokens = self._enc.encode(text)
            stride = self.target_size - self.overlap
            chunks = []
            i = 0
            while i < len(tokens):
                window = tokens[i: i + self.target_size]
                chunks.append(self._enc.decode(window))
                i += stride
            return chunks

        # Greedily merge sentences into chunks of target_size
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens > self.target_size and current:
                chunks.append(" ".join(current))
                # Keep overlap: take sentences from the end of current
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for s in reversed(current):
                    st = self._count_tokens(s)
                    if overlap_tokens + st > self.overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += st
                current = overlap_sentences
                current_tokens = overlap_tokens

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def chunk(self, doc: dict[str, Any]) -> list[ChunkRecord]:
        text = doc.get("cleaned_text", "")
        if not text.strip():
            return []

        # Split on paragraph boundaries
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Merge small paragraphs, split large ones
        merged_chunks: list[tuple[str, str]] = []  # (heading, text)
        current_heading = ""
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            # Check if paragraph starts with a heading
            heading = self._detect_heading(para)
            if heading:
                current_heading = heading

            para_tokens = self._count_tokens(para)

            if para_tokens > self.target_size:
                # Flush current buffer first
                if current_parts:
                    merged_chunks.append((current_heading, "\n\n".join(current_parts)))
                    current_parts = []
                    current_tokens = 0

                # Sub-split the oversized paragraph
                sub_chunks = self._sub_split(para)
                for sc in sub_chunks:
                    merged_chunks.append((current_heading, sc))
            elif current_tokens + para_tokens > self.target_size and current_parts:
                # Flush current buffer
                merged_chunks.append((current_heading, "\n\n".join(current_parts)))
                current_parts = [para]
                current_tokens = para_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        # Flush remaining
        if current_parts:
            merged_chunks.append((current_heading, "\n\n".join(current_parts)))

        # Build ChunkRecords
        records: list[ChunkRecord] = []
        cursor = 0

        for idx, (heading, chunk_text) in enumerate(merged_chunks):
            char_start = text.find(chunk_text[:40], cursor) if len(chunk_text) >= 40 else text.find(chunk_text, cursor)
            if char_start == -1:
                char_start = cursor
            char_end = char_start + len(chunk_text)
            cursor = max(cursor, char_start)

            records.append(ChunkRecord(
                chunk_id=str(uuid.uuid4()),
                document_id=doc["document_id"],
                doc_source=doc["doc_source"],
                document_type=doc["document_type"],
                document_date=doc["document_date"],
                octus_company_id=doc["octus_company_id"],
                company_name=doc["company_name"],
                company_ids=doc["company_ids"],
                section_title=heading,
                chunk_index=idx,
                char_start=char_start,
                char_end=char_end,
                chunker_id=self.chunker_id,
                text=chunk_text,
            ))

        return records
