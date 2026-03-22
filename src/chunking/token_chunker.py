"""
Token-window chunker (sliding window).

Uses tiktoken to count tokens. Splits text into overlapping windows of
chunk_size tokens with chunk_overlap tokens of overlap.

Rationale (see docs/decisions.md):
  Token-window chunking is the robust baseline when document structure is
  inconsistent. It guarantees bounded chunk sizes and is easy to tune.
"""

from __future__ import annotations

import uuid
from typing import Any

import tiktoken

from src.chunking.base import BaseChunker, ChunkRecord

# Default encoding — cl100k_base covers GPT-4 and text-embedding-3-* models
_DEFAULT_ENCODING = "cl100k_base"


class TokenChunker(BaseChunker):
    """
    Sliding-window token chunker.

    Args:
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Tokens of overlap between consecutive chunks.
        encoding_name: tiktoken encoding name.
    """

    chunker_id = "token"

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = _DEFAULT_ENCODING,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, doc: dict[str, Any]) -> list[ChunkRecord]:
        text = doc.get("cleaned_text", "")
        if not text.strip():
            return []

        tokens = self._enc.encode(text)
        stride = self.chunk_size - self.chunk_overlap
        chunks: list[ChunkRecord] = []
        chunk_index = 0

        i = 0
        while i < len(tokens):
            window = tokens[i : i + self.chunk_size]
            chunk_text = self._enc.decode(window)

            # Approximate char positions by finding first occurrence of start/end
            # (decode may not reproduce exact offsets, so we search forward)
            start_search = 0 if chunk_index == 0 else max(0, chunks[-1].char_start)
            char_start = text.find(chunk_text[:20], start_search)  # anchor on first 20 chars
            if char_start == -1:
                char_start = 0
            char_end = char_start + len(chunk_text)

            chunks.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    document_id=doc["document_id"],
                    doc_source=doc["doc_source"],
                    document_type=doc["document_type"],
                    document_date=doc["document_date"],
                    octus_company_id=doc["octus_company_id"],
                    company_name=doc["company_name"],
                    company_ids=doc["company_ids"],
                    section_title="",  # no structural title for token chunks
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                    chunker_id=self.chunker_id,
                    text=chunk_text,
                )
            )

            i += stride
            chunk_index += 1

        return chunks
