"""
Hybrid chunker.

Routes documents to the most appropriate chunker based on document type:
  - SEC filings → SECSectionChunker (structure-aware)
  - Transcripts → HeadingChunker (speaker/section-aware)
  - Fallback → TokenChunker (robust baseline)

Rationale:
  Transcripts and SEC filings have fundamentally different structures.
  A one-size-fits-all chunker sacrifices quality on at least one type.
  The hybrid approach delegates to the best chunker per document type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.chunking.base import BaseChunker, ChunkRecord
from src.chunking.heading_chunker import HeadingChunker
from src.chunking.sec_section_chunker import SECSectionChunker
from src.chunking.token_chunker import TokenChunker


class HybridChunker(BaseChunker):
    """
    Routes to the best chunker based on doc_source.

    Args:
        chunk_size: Token window size for TokenChunker fallback.
        chunk_overlap: Overlap for TokenChunker fallback.
        max_section_tokens: Max section size for SECSectionChunker.
    """

    chunker_id = "hybrid"

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        max_section_tokens: int = 2048,
    ) -> None:
        self._sec = SECSectionChunker(max_section_tokens=max_section_tokens)
        self._heading = HeadingChunker()
        self._token = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, doc: dict[str, Any]) -> list[ChunkRecord]:
        doc_source = doc.get("doc_source", "")
        raw_path = doc.get("raw_path", "")

        if doc_source == "sec_filing" and raw_path and Path(raw_path).exists():
            chunks = self._sec.chunk(doc)
        elif doc_source == "transcript":
            chunks = self._heading.chunk(doc)
        else:
            chunks = self._token.chunk(doc)

        # Override chunker_id to "hybrid" for consistency
        for c in chunks:
            c.chunker_id = self.chunker_id

        return chunks
