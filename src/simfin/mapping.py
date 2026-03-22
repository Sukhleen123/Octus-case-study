"""
Octus -> SimFin company mapping pipeline.

Uses rapidfuzz to fuzzy-match Octus company names to SimFin company names/tickers.
Writes company_map.{parquet|csv} with match status and scores.

References:
  - SimFin Python API docs: https://simfin.readthedocs.io/
  - SimFin GitHub: https://github.com/SimFin/simfin
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.simfin.storage import read_table, write_table

logger = logging.getLogger(__name__)

# Score threshold for auto_matched status
_AUTO_MATCH_THRESHOLD = 90.0

# Patterns stripped before fuzzy matching so that legal suffixes and noise
# (e.g. "INC /DE/", "CORP.", "LLC") don't penalise otherwise-identical names.
_LEGAL_SUFFIXES = re.compile(
    r"\b(incorporated|corporation|holdings?|enterprises?|international|"
    r"industries|group|inc|corp|co|ltd|llc|plc|lp|de)\b",
    re.IGNORECASE,
)
_NOISE_CHARS = re.compile(r"[/\\.,;:()'\"&]")


def _normalize_name(name: str) -> str:
    """Lowercase, strip legal suffixes and punctuation for fuzzy matching."""
    name = _NOISE_CHARS.sub(" ", name)
    name = _LEGAL_SUFFIXES.sub(" ", name)
    return " ".join(name.lower().split())


def _load_simfin_companies(settings: Any) -> pd.DataFrame:
    """
    Load SimFin company list via the simfin Python package.

    Falls back gracefully if the package is not installed or the API key is missing.
    """
    try:
        import simfin as sf

        if settings.simfin_api_key:
            sf.set_api_key(settings.simfin_api_key)

        cache_dir = str(Path(settings.cache_dir) / "simfin_py")
        sf.set_data_dir(cache_dir)

        df = sf.load_companies(market="us")
        logger.info("Loaded %d SimFin companies via simfin package", len(df))
        return df
    except Exception as e:
        logger.warning("Could not load SimFin companies via package: %s", e)
        return pd.DataFrame()


def _find_name_col(df: pd.DataFrame) -> str | None:
    """Find the company name column in a SimFin companies DataFrame."""
    for col in ["Company Name", "Name", "company_name", "companyName"]:
        if col in df.columns:
            return col
    return None


def build_company_map(settings: Any) -> pd.DataFrame:
    """
    Build and write company_map.{fmt}.

    Strategy:
    1. Load Octus company names from company_metadata.json
    2. Load SimFin company list
    3. Fuzzy match each Octus company to SimFin companies
    4. Auto-promote matches with score >= 90 if settings.auto_promote_matched

    Returns the mapping DataFrame.
    """
    from rapidfuzz import fuzz, process

    raw_dir = Path(settings.octus_raw_dir)
    out_dir = Path(
        settings.simfin_processed_path
        if hasattr(settings, "simfin_processed_path")
        else Path(settings.processed_dir) / "simfin"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = settings.table_format

    companies = json.loads(
        (raw_dir / "company_metadata.json").read_text(encoding="utf-8")
    )
    octus_df = pd.DataFrame(companies)[["octus_company_id", "company_name", "sub_industry"]].copy()

    simfin_df = _load_simfin_companies(settings)
    name_col = _find_name_col(simfin_df)

    rows = []
    now = datetime.utcnow().isoformat()

    if simfin_df.empty or name_col is None:
        logger.warning("SimFin company list unavailable — creating stub mapping")
        for _, r in octus_df.iterrows():
            rows.append({
                "octus_company_id": r["octus_company_id"],
                "company_name": r["company_name"],
                "suggested_ticker": "",
                "suggested_simfin_id": None,
                "match_score": 0.0,
                "status": "needs_review",
                "updated_at": now,
            })
    else:
        choices = simfin_df[name_col].astype(str).tolist()
        tickers = simfin_df.index.astype(str).tolist()

        for _, r in octus_df.iterrows():
            query = str(r["company_name"])

            # Log top candidates so the mapping can be audited
            top = process.extract(
                query, choices, scorer=fuzz.WRatio,
                processor=_normalize_name, limit=3,
            )
            logger.debug(
                "SimFin match candidates for '%s': %s",
                query,
                [(m[0], round(m[1], 1)) for m in top],
            )

            match = process.extractOne(
                query, choices, scorer=fuzz.WRatio, processor=_normalize_name
            )
            if match:
                matched_name, score, idx = match
                ticker = tickers[idx] if idx < len(tickers) else ""
                if score >= _AUTO_MATCH_THRESHOLD:
                    status = "confirmed" if settings.auto_promote_matched else "auto_matched"
                else:
                    status = "needs_review"
            else:
                ticker = ""
                score = 0.0
                status = "needs_review"

            rows.append({
                "octus_company_id": r["octus_company_id"],
                "company_name": r["company_name"],
                "suggested_ticker": ticker,
                "suggested_simfin_id": None,
                "match_score": float(score),
                "status": status,
                "updated_at": now,
            })

    mapping_df = pd.DataFrame(rows)
    out_path = out_dir / f"company_map.{fmt}"
    write_table(mapping_df, out_path, fmt)
    logger.info("Company map written to %s (%d rows)", out_path, len(mapping_df))
    return mapping_df


def filter_by_mapping_mode(mapping_df: pd.DataFrame, mapping_mode: str) -> pd.DataFrame:
    """
    Filter company_map rows by mapping_mode setting.

    mapping_mode:
      "confirmed"    → only status == "confirmed"
      "auto_matched" → only status == "auto_matched"
      "both"         → confirmed + auto_matched
    """
    if mapping_mode == "confirmed":
        return mapping_df[mapping_df["status"] == "confirmed"]
    elif mapping_mode == "auto_matched":
        return mapping_df[mapping_df["status"] == "auto_matched"]
    elif mapping_mode == "both":
        return mapping_df[mapping_df["status"].isin(["confirmed", "auto_matched"])]
    return mapping_df
