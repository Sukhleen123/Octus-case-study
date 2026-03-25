"""
SEC-section chunker.

Parses SEC filing HTML to split on actual filing section boundaries
(PART I/II, Item 1, Item 1A, Risk Factors, MD&A, etc.), preserving
table structure as markdown.

For transcript documents, falls back to heading chunker behavior.

Rationale:
  SEC filings have a standardized structure with clearly demarcated
  sections. Chunking on these boundaries preserves semantic coherence
  far better than arbitrary token windows, and keeps tables intact
  for financial data retrieval.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import tiktoken
from bs4 import BeautifulSoup, Tag

from src.chunking.base import BaseChunker, ChunkRecord

_DEFAULT_ENCODING = "cl100k_base"

# Regex patterns for SEC section boundaries (applied to extracted text).
# Matches: "ITEM 1.", "Item 1A.", "PART I.", "PART II:", etc.
_SEC_SECTION_RE = re.compile(
    r"^[\s]*("
    r"(?:PART|Part)\s+(?:I{1,3}|IV|[1-4])[\s.:]*(?:FINANCIAL\s+INFORMATION|OTHER\s+INFORMATION)?"
    r"|"
    r"(?:ITEM|Item)\s+\d+[A-Za-z]?\.?\s*[.\u2014\u2013\-]?\s*[A-Z].*"
    r")$",
    re.MULTILINE,
)

# 10-K item titles (annual report structure)
_10K_ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1a": "Risk Factors",
    "1b": "Unresolved Staff Comments",
    "1c": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market Information",
    "6": "Selected Financial Data",
    "7": "MD&A",
    "7a": "Quantitative and Qualitative Disclosures",
    "8": "Financial Statements",
    "9": "Changes in and Disagreements with Accountants",
    "9a": "Controls and Procedures",
    "9b": "Other Information",
}

# 10-Q Part I item titles (quarterly report — Part I)
_10Q_PART1_TITLES: dict[str, str] = {
    "1": "Financial Statements",
    "2": "MD&A",
    "3": "Quantitative and Qualitative Disclosures",
    "4": "Controls and Procedures",
}

# 10-Q Part II item titles (quarterly report — Part II)
_10Q_PART2_TITLES: dict[str, str] = {
    "1": "Legal Proceedings",
    "1a": "Risk Factors",
    "2": "Unregistered Sales of Equity Securities",
    "3": "Defaults Upon Senior Securities",
    "4": "Mine Safety Disclosures",
    "5": "Other Information",
    "6": "Exhibits",
}


def _get_item_titles(document_type: str, current_part: str) -> dict[str, str]:
    """Return the correct item-title mapping for the given filing type and part."""
    dt = (document_type or "").upper()
    if dt == "10-Q":
        return _10Q_PART2_TITLES if current_part == "II" else _10Q_PART1_TITLES
    return _10K_ITEM_TITLES


def _table_to_markdown(table_tag: Tag) -> str:
    """Convert an HTML <table> to a markdown table."""
    rows = table_tag.find_all("tr")
    if not rows:
        return ""

    md_rows: list[list[str]] = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        cell_texts = [c.get_text(strip=True).replace("|", "/") for c in cells]
        if any(cell_texts):
            md_rows.append(cell_texts)

    if not md_rows:
        return ""

    # Normalize column counts
    max_cols = max(len(r) for r in md_rows)
    for r in md_rows:
        while len(r) < max_cols:
            r.append("")

    lines = []
    lines.append("| " + " | ".join(md_rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in md_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_sec_text(html: str) -> str:
    """
    Extract text from SEC HTML, converting tables to markdown.

    Unlike the generic html_to_text.extract_text(), this preserves
    table structure as markdown and strips hidden XBRL headers.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove hidden XBRL header elements
    for tag in soup.find_all(style=lambda s: s and "display:none" in s.lower() if s else False):
        tag.decompose()
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    # Remove ix:header elements (XBRL inline headers)
    for tag in soup.find_all("ix:header"):
        tag.decompose()

    # Convert tables to markdown before extracting text
    for table in soup.find_all("table"):
        md = _table_to_markdown(table)
        if md:
            table.replace_with(soup.new_string(f"\n\n{md}\n\n"))
        else:
            table.decompose()

    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = text.splitlines()
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
            prev_blank = False
        else:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True

    return "\n".join(cleaned).strip()


_PART_RE = re.compile(
    r"^[\s]*(?:PART|Part)\s+(I{1,3}|IV|[1-4])\b",
    re.MULTILINE,
)


def _split_sec_sections(text: str) -> list[tuple[str, str, str]]:
    """
    Split SEC filing text into (section_title, section_text, current_part) triples.

    Tracks PART I / PART II transitions so 10-Q items can be titled correctly.
    The first block before any section header (typically table of contents) gets title "".
    """
    sections: list[tuple[str, str, str]] = []
    matches = list(_SEC_SECTION_RE.finditer(text))

    if not matches:
        return [("", text.strip(), "I")]

    # Content before first section header
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble, "I"))

    current_part = "I"
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        # Update current_part when we hit a PART boundary
        part_match = _PART_RE.match(title)
        if part_match:
            roman = part_match.group(1).upper()
            # Normalize roman numerals to "I" or "II" (10-Q only has two parts)
            current_part = "II" if roman in ("II", "2") else "I"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body, current_part))

    return sections


def _normalize_section_title(raw_title: str, item_titles: dict[str, str]) -> str:
    """Extract a clean section title from the raw matched text."""
    item_match = re.match(r"(?:ITEM|Item)\s+(\d+[A-Za-z]?)", raw_title, re.IGNORECASE)
    if item_match:
        item_key = item_match.group(1).lower()
        known = item_titles.get(item_key)
        if known:
            return f"Item {item_match.group(1).upper()} - {known}"
    return raw_title


class SECSectionChunker(BaseChunker):
    """
    Splits SEC filings on actual section boundaries (PART/ITEM).

    Preserves financial tables as markdown. Falls back to heading-based
    splitting for transcript documents.

    Args:
        max_section_tokens: If a section exceeds this, sub-split with
            token-window chunking within the section.
    """

    chunker_id = "sec_section"

    def __init__(
        self,
        max_section_tokens: int = 2048,
        sub_chunk_overlap: int = 64,
    ) -> None:
        self.max_section_tokens = max_section_tokens
        self.sub_chunk_overlap = sub_chunk_overlap
        self._enc = tiktoken.get_encoding(_DEFAULT_ENCODING)

    def chunk(self, doc: dict[str, Any]) -> list[ChunkRecord]:
        # For SEC filings, try to read raw HTML for structure-aware extraction
        raw_path = doc.get("raw_path", "")
        if raw_path and Path(raw_path).exists():
            with open(raw_path, encoding="utf-8", errors="replace") as f:
                html = f.read()
            text = _extract_sec_text(html)
        else:
            text = doc.get("cleaned_text", "")

        if not text.strip():
            return []

        sections = _split_sec_sections(text)
        chunks: list[ChunkRecord] = []
        chunk_index = 0
        document_type = doc.get("document_type", "")

        for title, section_text, current_part in sections:
            item_titles = _get_item_titles(document_type, current_part)
            normalized_title = _normalize_section_title(title, item_titles) if title else ""
            tokens = self._enc.encode(section_text)

            if len(tokens) <= self.max_section_tokens:
                chunks.append(self.make_record(
                    doc, normalized_title, section_text, chunk_index, text,
                ))
                chunk_index += 1
            else:
                # Sub-split large sections with token windows
                stride = self.max_section_tokens - self.sub_chunk_overlap
                i = 0
                sub_idx = 0
                while i < len(tokens):
                    window = tokens[i: i + self.max_section_tokens]
                    chunk_text = self._enc.decode(window)
                    sub_title = f"{normalized_title} (part {sub_idx + 1})" if normalized_title else ""
                    chunks.append(self.make_record(
                        doc, sub_title, chunk_text, chunk_index, text,
                    ))
                    chunk_index += 1
                    sub_idx += 1
                    i += stride

        return chunks

    def make_record(
        self,
        doc: dict[str, Any],
        title: str,
        text: str,
        index: int,
        full_text: str,
    ) -> ChunkRecord:
        # Approximate char positions
        char_start = full_text.find(text[:50]) if len(text) >= 50 else full_text.find(text)
        if char_start == -1:
            char_start = 0
        char_end = char_start + len(text)

        return ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            document_id=doc["document_id"],
            doc_source=doc["doc_source"],
            document_type=doc["document_type"],
            document_date=doc["document_date"],
            octus_company_id=doc["octus_company_id"],
            company_name=doc["company_name"],
            company_ids=doc["company_ids"],
            section_title=title,
            chunk_index=index,
            char_start=char_start,
            char_end=char_end,
            chunker_id=self.chunker_id,
            text=text,
        )
