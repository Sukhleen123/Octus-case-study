"""Base class for all chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ChunkRecord:
    """
    A single chunk produced by a chunker.

    Every field is required — no Optional here — because downstream code
    (vectorstore upsert, experiment comparison) depends on all fields being present.
    """
    chunk_id: str           # UUID
    document_id: str
    doc_source: str         # "transcript" | "sec_filing"
    document_type: str
    document_date: datetime
    octus_company_id: str
    company_name: str
    company_ids: str        # JSON-encoded list[int]
    section_title: str      # heading text, or "" if unknown
    chunk_index: int        # 0-based index within document
    char_start: int
    char_end: int
    chunker_id: str         # "heading" | "token"
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "doc_source": self.doc_source,
            "document_type": self.document_type,
            "document_date": self.document_date,
            "octus_company_id": self.octus_company_id,
            "company_name": self.company_name,
            "company_ids": self.company_ids,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chunker_id": self.chunker_id,
            "text": self.text,
        }


class BaseChunker(ABC):
    """Abstract chunker interface."""

    chunker_id: str  # must be set by subclasses

    @abstractmethod
    def chunk(self, doc: dict[str, Any]) -> list[ChunkRecord]:
        """
        Split a document into chunks.

        Args:
            doc: An OctusDocument dict (as stored in documents.parquet).

        Returns:
            List of ChunkRecord objects.
        """
        ...
