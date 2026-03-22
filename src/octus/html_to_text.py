"""
Extract clean plain text from HTML strings.

Used for both transcript body (inline HTML) and SEC filing HTML files.
"""

from __future__ import annotations

from bs4 import BeautifulSoup


def extract_text(html: str) -> str:
    """
    Parse *html* and return cleaned plain text.

    Strategy:
    - Use lxml parser for speed and robustness
    - Remove <script> and <style> tags entirely
    - Use get_text with newline separator to preserve paragraph breaks
    - Collapse excess whitespace while keeping single blank lines
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noise elements
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Normalize whitespace: collapse multiple blank lines to one
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
            prev_blank = False
        else:
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True

    return "\n".join(cleaned_lines).strip()


def extract_text_from_file(path: str) -> str:
    """Read an HTML file and return cleaned plain text."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return extract_text(f.read())
