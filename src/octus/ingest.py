"""
Octus data ingestion pipeline.

run_ingestion(settings) is the entry point. It:
  1. Loads company_metadata.json -> builds entity_id -> (octus_company_id, company_name) map
  2. Loads transcripts.json -> normalizes -> OctusDocument records
  3. Loads sec_filings_metadata.json -> verifies HTML exists -> deduplicates -> OctusDocument records
  4. Writes data/processed/octus/documents.{fmt}
  5. Writes data/processed/octus/ingest_report.{fmt}

Only fields actually present in the raw JSON files are referenced here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.settings import Settings
from src.octus.dedupe import dedupe_sec_filings
from src.octus.html_to_text import extract_text, extract_text_from_file
from src.octus.normalize import parse_company_ids, parse_sec_date, parse_transcript_date
from src.simfin.storage import write_table

logger = logging.getLogger(__name__)


# ── Entity map ─────────────────────────────────────────────────────────────────

def build_entity_map(
    companies: list[dict[str, Any]]
) -> dict[int, tuple[str, str]]:
    """
    Build a mapping: entity_id (int) -> (octus_company_id, company_name).

    company_ids is a comma-separated string in the raw data.
    Logs a warning on collision (same entity_id mapped to multiple companies).
    """
    entity_map: dict[int, tuple[str, str]] = {}
    collisions: dict[int, set[str]] = {}

    for row in companies:
        octus_id = row["octus_company_id"]
        name = row["company_name"]
        ids = parse_company_ids(row.get("company_ids", ""))
        for cid in ids:
            if cid in entity_map and entity_map[cid][1] != name:
                collisions.setdefault(cid, {entity_map[cid][1]}).add(name)
            else:
                entity_map[cid] = (octus_id, name)

    if collisions:
        logger.warning("Entity ID collisions detected: %s", collisions)

    logger.info("Entity map built: %d unique entity IDs", len(entity_map))
    return entity_map


def _resolve_company(
    entity_ids: list[int],
    entity_map: dict[int, tuple[str, str]],
) -> tuple[str, str, list[int]]:
    """
    Resolve octus_company_id and company_name from a list of entity IDs.

    Returns (octus_company_id, company_name, entity_ids).
    Uses the first successfully mapped ID. Logs a warning if unmapped.
    """
    for eid in entity_ids:
        if eid in entity_map:
            octus_id, name = entity_map[eid]
            return octus_id, name, entity_ids
    logger.warning("Could not resolve entity IDs %s to any company", entity_ids)
    return "", "UNKNOWN", entity_ids


# ── Transcript ingestion ───────────────────────────────────────────────────────

def ingest_transcripts(
    transcripts: list[dict[str, Any]],
    entity_map: dict[int, tuple[str, str]],
) -> list[dict[str, Any]]:
    """Normalize transcript records into OctusDocument dicts."""
    docs = []
    for rec in transcripts:
        company_id_int = rec["company_id"]  # single int
        octus_id, company_name, company_ids = _resolve_company(
            [company_id_int], entity_map
        )
        cleaned_text = extract_text(rec["body"])
        doc = {
            "doc_source": "transcript",
            "document_id": rec["document_id"],
            "source_type": rec["source_type"],
            "document_type": rec["document_type"],
            "document_date": parse_transcript_date(rec["document_date"]),
            "octus_company_id": octus_id,
            "company_name": company_name,
            "company_ids": json.dumps(company_ids),
            "raw_path": "",  # transcript body is inline
            "cleaned_text": cleaned_text,
        }
        docs.append(doc)
    logger.info("Ingested %d transcripts", len(docs))
    return docs


# ── SEC filing ingestion ───────────────────────────────────────────────────────

def ingest_sec_filings(
    filings: list[dict[str, Any]],
    entity_map: dict[int, tuple[str, str]],
    sec_html_dir: Path,
    require_html: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Normalize SEC filing records into OctusDocument dicts.

    Verifies HTML file existence and deduplicates before normalization.

    Returns:
        (canonical_docs, duplicate_records)
    """
    # ── Verify HTML presence ─────────────────────────────────────────────────
    missing = []
    for rec in filings:
        html_path = sec_html_dir / f"{rec['document_id']}.html"
        if not html_path.exists():
            missing.append(rec["document_id"])

    if missing:
        msg = f"Missing SEC HTML files for {len(missing)} filing(s): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        if require_html:
            raise FileNotFoundError(msg)
        else:
            logger.warning(msg)

    # ── Deduplication ────────────────────────────────────────────────────────
    canonical_recs, duplicate_recs = dedupe_sec_filings(filings)
    logger.info(
        "SEC filings: %d total, %d canonical, %d duplicates",
        len(filings), len(canonical_recs), len(duplicate_recs),
    )

    # ── Normalize canonical records ──────────────────────────────────────────
    docs = []
    for rec in canonical_recs:
        company_ids_list = rec["company_id"]  # already list[int]
        octus_id, company_name, _ = _resolve_company(company_ids_list, entity_map)

        html_path = sec_html_dir / f"{rec['document_id']}.html"
        if html_path.exists():
            cleaned_text = extract_text_from_file(str(html_path))
        else:
            cleaned_text = ""

        doc = {
            "doc_source": "sec_filing",
            "document_id": rec["document_id"],
            "source_type": rec["source_type"],
            "document_type": rec["document_type"],
            "document_date": parse_sec_date(rec["document_date"]),
            "octus_company_id": octus_id,
            "company_name": company_name,
            "company_ids": json.dumps(company_ids_list),
            "raw_path": str(html_path),
            "cleaned_text": cleaned_text,
        }
        docs.append(doc)

    return docs, duplicate_recs


# ── Main entry point ───────────────────────────────────────────────────────────

def run_ingestion(settings: Settings) -> pd.DataFrame:
    """
    Full Octus ingestion pipeline.

    Reads raw JSON files, normalizes, deduplicates, writes:
      - data/processed/octus/documents.{fmt}
      - data/processed/octus/ingest_report.{fmt}

    Returns the documents DataFrame.
    """
    raw_dir = Path(settings.octus_raw_dir)
    sec_html_dir = raw_dir / "sec_html"
    out_dir = Path(settings.octus_processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = settings.table_format

    # ── Load raw JSON ────────────────────────────────────────────────────────
    companies: list[dict] = json.loads(
        (raw_dir / "company_metadata.json").read_text(encoding="utf-8")
    )
    transcripts: list[dict] = json.loads(
        (raw_dir / "transcripts.json").read_text(encoding="utf-8")
    )
    sec_filings: list[dict] = json.loads(
        (raw_dir / "sec_filings_metadata.json").read_text(encoding="utf-8")
    )

    logger.info(
        "Loaded raw: %d companies, %d transcripts, %d sec_filings",
        len(companies), len(transcripts), len(sec_filings),
    )

    # ── Build entity map ────────────────────────────────────────────────────
    entity_map = build_entity_map(companies)

    # ── Ingest ──────────────────────────────────────────────────────────────
    transcript_docs = ingest_transcripts(transcripts, entity_map)
    sec_docs, duplicate_recs = ingest_sec_filings(
        sec_filings,
        entity_map,
        sec_html_dir,
        require_html=settings.require_sec_html,
    )

    all_docs = transcript_docs + sec_docs

    # ── Write documents ──────────────────────────────────────────────────────
    docs_df = pd.DataFrame(all_docs)
    docs_path = out_dir / f"documents.{fmt}"
    write_table(docs_df, docs_path, fmt)
    logger.info("Wrote %d documents to %s", len(docs_df), docs_path)

    # ── Write ingest report ──────────────────────────────────────────────────
    report_rows = []
    for dup in duplicate_recs:
        report_rows.append({
            "document_id": dup.get("document_id", ""),
            "document_type": dup.get("document_type", ""),
            "document_date": dup.get("document_date", ""),
            "company_id": json.dumps(dup.get("company_id", [])),
            "canonical_id": dup.get("_canonical_id", ""),
            "reason": dup.get("_dedupe_reason", ""),
        })

    report_df = pd.DataFrame(report_rows)
    report_path = out_dir / f"ingest_report.{fmt}"
    write_table(report_df, report_path, fmt)
    logger.info(
        "Ingest report: %d duplicate entries written to %s", len(report_df), report_path
    )

    return docs_df
