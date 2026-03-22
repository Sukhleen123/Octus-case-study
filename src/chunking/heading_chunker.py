"""
Heading-aware chunker.

Splits document text on HTML headings (<h1>-<h6>) or Markdown headers (## ...).
Each section becomes one chunk, preserving the section title for citations.

Rationale (see docs/decisions.md):
  Heading-based chunking preserves document structure and improves citation
  readability — chunk boundaries align with meaningful semantic units.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from src.chunking.base import BaseChunker, ChunkRecord


# Regex to detect headings in two formats:
#   1. Markdown: "## Section Title"
#   2. Inline HTML-stripped: lines that appear to be headings (ALL CAPS or title-cased short lines)
#      We also detect patterns left by HTML-stripping like "RISK FACTORS\n" in SEC docs.
_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,6}\s+(?P<md_title>.+)"         # Markdown ## headings
    r"|"
    r"(?P<html_title>[A-Z][A-Z\s,.\-&]{3,60})$"  # ALL-CAPS heading (common in SEC filings)
    r")$",
    re.MULTILINE,
)


def _split_on_headings(text: str) -> list[tuple[str, str]]:
    """
    Split *text* into (section_title, section_text) pairs.

    The first section before any heading gets the title "".
    """
    sections: list[tuple[str, str]] = []
    pos = 0
    current_title = ""
    current_start = 0

    for m in _HEADING_RE.finditer(text):
        # Save preceding section
        section_text = text[current_start:m.start()].strip()
        if section_text:
            sections.append((current_title, section_text))
        current_title = (m.group("md_title") or m.group("html_title") or "").strip()
        current_start = m.end()
        pos = m.start()

    # Remainder after last heading
    remainder = text[current_start:].strip()
    if remainder:
        sections.append((current_title, remainder))

    # If no headings found, treat entire doc as one section
    if not sections:
        sections = [("", text.strip())]

    return sections


class HeadingChunker(BaseChunker):
    """
    Splits cleaned_text into one chunk per section (delimited by headings).

    Chunks can be large if sections are long; that is acceptable because
    heading-based chunking prioritizes structural coherence over uniform size.
    """

    chunker_id = "heading"

    def chunk(self, doc: dict[str, Any]) -> list[ChunkRecord]:
        text = doc.get("cleaned_text", "")
        if not text.strip():
            return []

        sections = _split_on_headings(text)
        chunks: list[ChunkRecord] = []
        cursor = 0

        for idx, (title, section_text) in enumerate(sections):
            # Find char_start / char_end in the original text
            start = text.find(section_text, cursor)
            if start == -1:
                start = cursor
            end = start + len(section_text)
            cursor = end

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
                    section_title=title,
                    chunk_index=idx,
                    char_start=start,
                    char_end=end,
                    chunker_id=self.chunker_id,
                    text=section_text,
                )
            )

        return chunks
