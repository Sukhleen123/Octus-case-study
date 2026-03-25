"""
Shared utility functions for the agent pipeline.
Logic extracted from the former Orchestrator class.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.consts import DOC_TYPE_KEYWORDS, HYDE_TRIGGERS, SEC_KEYWORDS

logger = logging.getLogger(__name__)


def extract_companies(query: str, company_map: Any) -> list[str]:
    """Return all company names found in the query via substring and fuzzy matching."""
    if company_map is None:
        return []
    try:
        import pandas as pd
        from rapidfuzz import fuzz, process

        df = company_map
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []

        known_names = df["company_name"].dropna().unique().tolist()
        q_lower = query.lower()
        matched: list[str] = []

        for name in known_names:
            words = [w for w in name.lower().split() if len(w) > 4]
            if name.lower() in q_lower or (words and any(w in q_lower for w in words)):
                matched.append(name)

        if not matched:
            hits = process.extract(query, known_names, scorer=fuzz.partial_ratio, limit=3)
            matched = [name for name, score, _ in hits if score >= 70]

        for name in matched:
            logger.info("Extracted company from query: %s", name)
        return matched
    except Exception as e:
        logger.warning("Company extraction failed: %s", e)
    return []


def resolve_tickers(companies: list[str], company_map: Any) -> list[str]:
    """Look up SimFin tickers for all resolved companies."""
    if company_map is None or not companies:
        return []
    try:
        import pandas as pd

        df = company_map
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []

        mask = df["company_name"].str.lower().isin([c.lower() for c in companies])
        subset = df[mask]
        tickers = (
            subset["suggested_ticker"]
            .dropna()
            .loc[lambda s: s.str.strip() != ""]
            .unique()
            .tolist()
        )
        return tickers[:10]
    except Exception as e:
        logger.warning("Could not resolve tickers: %s", e)
    return []


def infer_doc_filters(
    query: str, companies: list[str], retrieval_kwargs: dict
) -> dict:
    """
    Infer Pinecone metadata filters from query keywords.
    Explicit retrieval_kwargs take precedence over inferred values.
    """
    filters: dict = {}
    q_lower = query.lower()

    for kw, doc_type in DOC_TYPE_KEYWORDS.items():
        if kw in q_lower:
            filters["document_type"] = doc_type
            break

    if "document_type" not in filters:
        if any(kw in q_lower for kw in SEC_KEYWORDS):
            filters["doc_source"] = "sec_filing"

    if len(companies) == 1:
        filters["company_name"] = companies[0]
    elif len(companies) > 1:
        filters["company_name"] = companies

    return {**filters, **retrieval_kwargs}


def maybe_expand_query(
    query: str,
    company: str | None,
    hyde: bool,
    llm_client: Any,
    model: str,
) -> str:
    """
    Replace the retrieval query with a hypothetical ideal passage (HyDE).
    Only fires when hyde=True and the query matches forward-looking patterns.
    """
    if not hyde or not HYDE_TRIGGERS.search(query):
        return query
    if llm_client is None:
        return query

    company_context = f" for {company}" if company else ""
    prompt = (
        f"Write a 2-3 sentence passage from a financial filing or earnings call "
        f"that would be the ideal answer to: \"{query}\"{company_context}\n"
        f"Write only the passage itself. Use specific numbers, guidance ranges, "
        f"or management quotes typical of financial documents."
    )
    try:
        resp = llm_client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        expanded = resp.content[0].text.strip()
        logger.info("HyDE expansion: '%s...' → '%s...'", query[:40], expanded[:80])
        return expanded
    except Exception as e:
        logger.warning("HyDE expansion failed, using original query: %s", e)
        return query
