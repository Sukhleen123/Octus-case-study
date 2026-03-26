"""
SEC-section chunker.

Parses SEC filing HTML to split on actual filing section boundaries
(PART I/II, Item 1, Item 1A, Risk Factors, MD&A, etc.), preserving
table structure as markdown.

Two parsing strategies:
  1. TOC-anchor (preferred): when the document has <a href="#..."> TOC links
     pointing to semantic anchors (id="item_1_business", etc.), use those
     to split the DOM directly. Section titles come from the TOC text, so
     company-specific names like "Item 2. MD&A" are preserved accurately.
  2. Regex-text (fallback): for Wdesk-generated or otherwise anchor-less filings,
     extract text and split on PART/ITEM boundary patterns.

For transcript documents, falls back to heading chunker behavior.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import tiktoken
from bs4 import BeautifulSoup, NavigableString, Tag

from src.chunking.base import BaseChunker, ChunkRecord

_DEFAULT_ENCODING = "cl100k_base"

# ── Regex patterns (fallback text-split path) ─────────────────────────────────

_SEC_SECTION_RE = re.compile(
    r"^[\s]*("
    r"(?:PART|Part)\s+(?:I{1,3}|IV|[1-4])[\s.:]*(?:FINANCIAL\s+INFORMATION|OTHER\s+INFORMATION)?"
    r"|"
    r"(?:ITEM|Item)\s+\d+[A-Za-z]?\.?\s*[.\u2014\u2013\-]?\s*[A-Z].*"
    r")$",
    re.MULTILINE,
)

_PART_RE = re.compile(
    r"^[\s]*(?:PART|Part)\s+(I{1,3}|IV|[1-4])\b",
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

_10Q_PART1_TITLES: dict[str, str] = {
    "1": "Financial Statements",
    "2": "MD&A",
    "3": "Quantitative and Qualitative Disclosures",
    "4": "Controls and Procedures",
}

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
    dt = (document_type or "").upper()
    if dt == "10-Q":
        return _10Q_PART2_TITLES if current_part == "II" else _10Q_PART1_TITLES
    return _10K_ITEM_TITLES


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _table_to_markdown(table_tag: Tag) -> str:
    """Convert an HTML <table> to a markdown table string."""
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

    max_cols = max(len(r) for r in md_rows)
    for r in md_rows:
        while len(r) < max_cols:
            r.append("")

    lines = ["| " + " | ".join(md_rows[0]) + " |",
             "| " + " | ".join(["---"] * max_cols) + " |"]
    for row in md_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _clean_soup(soup: BeautifulSoup) -> None:
    """Remove hidden, script, style, and XBRL elements from soup in place."""
    for tag in soup.find_all(
        style=lambda s: s and "display:none" in s.lower() if s else False
    ):
        tag.decompose()
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    for tag in soup.find_all("ix:header"):
        tag.decompose()


def _soup_to_text(soup: BeautifulSoup) -> str:
    """Extract plain text from an already-cleaned soup, converting tables to markdown."""
    for table in soup.find_all("table"):
        md = _table_to_markdown(table)
        if md:
            table.replace_with(soup.new_string(f"\n\n{md}\n\n"))
        else:
            table.decompose()

    text = soup.get_text(separator="\n")
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


# ── TOC-anchor parsing path ───────────────────────────────────────────────────

def _extract_toc_map(soup: BeautifulSoup) -> dict[str, str]:
    """
    Build an anchor_id → human-readable title map from TOC href links.

    Looks for <a href="#..."> elements whose text matches a PART/ITEM reference.
    Returns an empty dict if fewer than 3 semantic links are found (indicating
    the document likely has no machine-readable TOC).
    """
    toc: dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not href.startswith("#"):
            continue
        anchor_id = href[1:]
        if not anchor_id:
            continue

        text = a.get_text(separator=" ", strip=True)
        # Strip trailing page numbers (e.g., "Item 1. Business   42")
        text = re.sub(r"\s+\d+\s*$", "", text).strip().rstrip(":–—- ")
        if not re.search(r"(?:part|item)\s+\d", text, re.IGNORECASE):
            continue
        if len(text) < 4:
            continue

        toc[anchor_id] = text

    if len(toc) < 3:
        return {}

    # Validate: at least 3 anchors must actually exist in the DOM
    found = sum(
        1 for aid in toc
        if soup.find(id=aid) or soup.find("a", attrs={"name": aid})
    )
    return toc if found >= 3 else {}


def _split_by_dom_anchors(
    soup: BeautifulSoup,
    toc_map: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Walk the DOM, collecting text and splitting at elements whose id (or <a name>)
    is in toc_map. Tables are converted to markdown inline.

    Returns list of (title, body_text) pairs; the preamble before the first anchor
    gets title "".
    """
    sections: list[tuple[str, str]] = []
    state: dict = {"title": "", "parts": []}

    def flush() -> None:
        body = "\n".join(state["parts"]).strip()
        # Collapse runs of blank lines
        body = re.sub(r"\n{3,}", "\n\n", body)
        if body:
            sections.append((state["title"], body))
        state["parts"] = []

    def visit(el: Any) -> None:
        if isinstance(el, NavigableString):
            t = str(el).strip()
            if t:
                state["parts"].append(t)
            return
        if not isinstance(el, Tag):
            return

        # Detect section boundary: id= or <a name=>
        tag_id = el.get("id", "")
        if not tag_id and el.name == "a":
            tag_id = el.get("name", "")

        if tag_id and tag_id in toc_map:
            flush()
            state["title"] = toc_map[tag_id]
            # Don't recurse into the anchor heading element — its text is the
            # title, not body content.
            return

        if el.name == "table":
            md = _table_to_markdown(el)
            if md:
                state["parts"].append(f"\n\n{md}\n\n")
            return  # don't recurse into table

        for child in el.children:
            visit(child)

    root = soup.body if soup.body else soup
    for child in root.children:
        visit(child)
    flush()

    return sections


def _part_from_title(title: str) -> str:
    """Return 'II' if the section title references Part II, else 'I'."""
    if re.search(r"\bpart\s+(?:ii|2)\b", title, re.IGNORECASE):
        return "II"
    return "I"


# ── Regex text-split path (fallback) ─────────────────────────────────────────

def _split_sec_sections(text: str) -> list[tuple[str, str, str]]:
    """
    Split SEC filing plain text into (section_title, section_text, current_part) triples.

    The first block before any section header (typically table of contents or
    cover page) gets an empty title.
    """
    sections: list[tuple[str, str, str]] = []
    matches = list(_SEC_SECTION_RE.finditer(text))

    if not matches:
        return [("", text.strip(), "I")]

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble, "I"))

    current_part = "I"
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        part_match = _PART_RE.match(title)
        if part_match:
            roman = part_match.group(1).upper()
            current_part = "II" if roman in ("II", "2") else "I"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body, current_part))

    return sections


def _normalize_section_title(raw_title: str, item_titles: dict[str, str]) -> str:
    """Map a raw regex-matched title to a clean human-readable name."""
    item_match = re.match(r"(?:ITEM|Item)\s+(\d+[A-Za-z]?)", raw_title, re.IGNORECASE)
    if item_match:
        item_key = item_match.group(1).lower()
        known = item_titles.get(item_key)
        if known:
            return f"Item {item_match.group(1).upper()} - {known}"
    return raw_title


# ── Chunker ───────────────────────────────────────────────────────────────────

class SECSectionChunker(BaseChunker):
    """
    Splits SEC filings on actual section boundaries (PART/ITEM).

    Parsing strategy (in priority order):
      1. TOC-anchor: extract section titles from TOC href links, split DOM at
         matching id= anchors. Preserves company-specific section names.
      2. Regex-text: extract plain text, split on PART/ITEM pattern matches.
         Used for Wdesk-generated or other anchor-less filings.

    Large sections are sub-split with token-window chunking.
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
        raw_path = doc.get("raw_path", "")
        document_type = doc.get("document_type", "")
        use_toc = False

        if raw_path and Path(raw_path).exists():
            with open(raw_path, encoding="utf-8", errors="replace") as f:
                html = f.read()

            soup = BeautifulSoup(html, "lxml")
            _clean_soup(soup)

            toc_map = _extract_toc_map(soup)
            if toc_map:
                raw_pairs = _split_by_dom_anchors(soup, toc_map)
                sections: list[tuple[str, str, str]] = [
                    (title, body, _part_from_title(title))
                    for title, body in raw_pairs
                ]
                use_toc = True
            else:
                text = _soup_to_text(soup)
                sections = _split_sec_sections(text)
        else:
            text = doc.get("cleaned_text", "")
            sections = _split_sec_sections(text)

        if not sections:
            return []

        chunks: list[ChunkRecord] = []
        chunk_index = 0

        for title, section_text, current_part in sections:
            if not section_text.strip():
                continue

            if use_toc:
                # TOC titles are already human-readable; use them directly
                normalized_title = title
            else:
                item_titles = _get_item_titles(document_type, current_part)
                normalized_title = _normalize_section_title(title, item_titles) if title else ""

            tokens = self._enc.encode(section_text)

            if len(tokens) <= self.max_section_tokens:
                chunks.append(self.make_record(
                    doc, normalized_title, section_text, chunk_index,
                ))
                chunk_index += 1
            else:
                stride = self.max_section_tokens - self.sub_chunk_overlap
                i = 0
                sub_idx = 0
                while i < len(tokens):
                    window = tokens[i: i + self.max_section_tokens]
                    chunk_text = self._enc.decode(window)
                    sub_title = (
                        f"{normalized_title} (part {sub_idx + 1})"
                        if normalized_title else ""
                    )
                    chunks.append(self.make_record(
                        doc, sub_title, chunk_text, chunk_index,
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
    ) -> ChunkRecord:
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
            char_start=0,
            char_end=len(text),
            chunker_id=self.chunker_id,
            text=text,
        )

