"""
Router node: decides which agents to invoke and prepares shared state.

Performs keyword-based routing, company/ticker extraction, doc filter
inference, and optional HyDE query expansion before any retrieval happens.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

from src.agents import runtime
from src.agents.consts import ALL_COMPANIES_TRIGGERS, DOC_TYPE_KEYWORDS, HYDE_TRIGGERS, OCTUS_KEYWORDS, SEC_KEYWORDS, SECTOR_TRIGGERS, SIMFIN_KEYWORDS
from src.agents.events import EventType
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

RouteDecision = Literal["octus", "simfin", "both"]


def is_all_companies_query(query: str) -> bool:
    """Return True if the query is asking about all companies in the database."""
    return bool(ALL_COMPANIES_TRIGGERS.search(query))


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


def extract_sector_companies(query: str) -> list[str]:
    """Match sector terms in query against company sub_industry labels in company_map.

    Only runs when the query contains explicit sector/industry framing words to avoid
    false positives from loose fuzzy matching on unrelated queries.
    """
    if not SECTOR_TRIGGERS.search(query):
        return []
    if runtime.company_map is None:
        return []
    try:
        import pandas as pd
        from rapidfuzz import fuzz, process

        df = runtime.company_map
        if not isinstance(df, pd.DataFrame) or df.empty or "sub_industry" not in df.columns:
            return []

        sub_industries = df["sub_industry"].dropna().loc[lambda s: s.str.strip() != ""].unique().tolist()
        if not sub_industries:
            return []

        q = query.lower()

        # Level 1: exact word match — any significant word from a sub_industry label in query
        for si in sub_industries:
            words = [w for w in si.lower().split() if len(w) > 4]
            if words and any(w in q for w in words):
                matched = df[df["sub_industry"] == si]["company_name"].tolist()
                if matched:
                    logger.info("Sector match (exact word) '%s' → %s", si, matched)
                    return matched

        # Level 2: fuzzy match query words against sub_industry labels
        query_words = [w for w in q.split() if len(w) > 4]
        for word in query_words:
            result = process.extractOne(word, sub_industries, scorer=fuzz.partial_ratio)
            if result and result[1] >= 75:
                si = result[0]
                matched = df[df["sub_industry"] == si]["company_name"].tolist()
                if matched:
                    logger.info("Sector match (fuzzy %.0f) '%s' → %s", result[1], si, matched)
                    return matched
    except Exception as e:
        logger.warning("Sector extraction failed: %s", e)
    return []


def infer_doc_filters(
    query: str, companies: list[str], retrieval_kwargs: dict
) -> dict:
    """Infer Pinecone metadata filters from query keywords. retrieval_kwargs take precedence."""
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
    """Replace query with a hypothetical ideal passage (HyDE) when enabled."""
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


def extract_doc_temporal_filters(query: str) -> dict:
    """Parse temporal intent from query for Pinecone document date filtering.

    Returns {"doc_date_from": str, "doc_date_to": str} (ISO date strings, "" = no filter).
    """
    today = date.today()
    q = query.lower()
    out: dict = {"doc_date_from": "", "doc_date_to": ""}

    m = re.search(r"(?:last|past)\s+(\d+)\s+quarters?", q)
    if m:
        n = int(m.group(1))
        out["doc_date_from"] = (today - timedelta(days=n * 91)).strftime("%Y-%m-%d")
        return out

    m = re.search(r"(?:last|past)\s+(\d+)\s+years?", q)
    if m:
        n = int(m.group(1))
        out["doc_date_from"] = date(today.year - n, today.month, today.day).strftime("%Y-%m-%d")
        return out

    m = re.search(r"(?:last|past)\s+(\d+)\s+months?", q)
    if m:
        n = int(m.group(1))
        out["doc_date_from"] = (today - timedelta(days=n * 30)).strftime("%Y-%m-%d")
        return out

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        yr = int(m.group(1))
        out["doc_date_from"] = f"{yr}-01-01"
        out["doc_date_to"] = f"{yr}-12-31"
        return out

    if re.search(r"\b(?:most\s+recent|latest)\b", q):
        out["doc_date_from"] = date(today.year - 1, today.month, today.day).strftime("%Y-%m-%d")
        return out

    if re.search(r"\brecent\b", q):
        out["doc_date_from"] = (today - timedelta(days=548)).strftime("%Y-%m-%d")
        return out

    # Broad context window for queries asking about history leading up to a recent metric
    if re.search(r"leading\s+up|shaped|context|background|historically", q):
        out["doc_date_from"] = date(today.year - 3, today.month, today.day).strftime("%Y-%m-%d")
        return out

    return out


def extract_simfin_temporal_filters(query: str) -> dict:
    """Parse temporal intent from query for SimFin DuckDB SQL-level filtering.

    Returns {"simfin_date_from": str, "simfin_max_quarterly_periods": int,
             "simfin_max_annual_periods": int}.
    """
    today = date.today()
    q = query.lower()
    out: dict = {
        "simfin_date_from": "",
        "simfin_max_quarterly_periods": 0,
        "simfin_max_annual_periods": 0,
    }

    m = re.search(r"(?:last|past)\s+(\d+)\s+quarters?", q)
    if m:
        out["simfin_max_quarterly_periods"] = int(m.group(1))
        return out

    m = re.search(r"(?:last|past)\s+(\d+)\s+years?", q)
    if m:
        n = int(m.group(1))
        out["simfin_max_annual_periods"] = n
        out["simfin_date_from"] = str(today.year - n)
        return out

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        out["simfin_date_from"] = m.group(1)
        return out

    if re.search(r"\b(?:most\s+recent|latest)\b", q):
        out["simfin_max_annual_periods"] = 1
        out["simfin_max_quarterly_periods"] = 4
        return out

    if re.search(r"\brecent\b", q):
        out["simfin_max_annual_periods"] = 2
        out["simfin_max_quarterly_periods"] = 6
        return out

    return out


def select_simfin_tables(query: str, all_companies_mode: bool) -> list[str]:
    """Return the minimal SimFin table set needed for this query.

    Returns [] (empty) to use all tables, or a specific subset.
    """
    q = query.lower()
    if all_companies_mode:
        return ["income_annual", "balance_annual"]
    if any(kw in q for kw in ("cash flow", "operating activities", "free cash")):
        return ["cashflow_annual", "cashflow_quarterly", "income_annual", "balance_annual"]
    if re.search(r"quarter(?:ly)?|q[1-4]\b", q):
        return ["income_quarterly", "balance_quarterly", "income_annual"]
    return []


def route_decision(query: str) -> RouteDecision:
    """
    Keyword-based routing: returns "octus", "simfin", or "both".
    Defaults to "both" when no keywords match (safe fallback).
    """
    q = query.lower()
    simfin_hits = sum(1 for kw in SIMFIN_KEYWORDS if kw in q)
    octus_hits = sum(1 for kw in OCTUS_KEYWORDS if kw in q)

    if simfin_hits > 0 and octus_hits > 0:
        return "both"
    if simfin_hits > octus_hits:
        return "simfin"
    if octus_hits > simfin_hits:
        return "octus"
    return "both"


def router_node(state: AgentState) -> dict:
    """
    LangGraph node: routes the query and prepares retrieval context.

    Sets: route, companies, tickers, doc_filters, doc_query, all_companies_mode, trace_events.
    """
    all_companies_mode = is_all_companies_query(state.query)

    sector_mode = False
    if all_companies_mode and runtime.company_map is not None:
        import pandas as pd
        df = runtime.company_map
        companies = df["company_name"].dropna().unique().tolist() if isinstance(df, pd.DataFrame) else []
    else:
        sector_companies = extract_sector_companies(state.query)
        if sector_companies:
            companies = sector_companies
            sector_mode = True
        else:
            companies = extract_companies(state.query, runtime.company_map)

    tickers = resolve_tickers(companies, runtime.company_map)
    decision = route_decision(state.query)

    # All-companies queries with financial data requests always hit both agents
    if all_companies_mode and tickers:
        decision = "both"

    # In all-companies or sector mode with multiple companies, omit company_name from
    # doc_filters — doc_agent applies per-company filters itself via parallel retrieval
    if all_companies_mode or (sector_mode and len(companies) > 1):
        doc_filters = infer_doc_filters(state.query, [], state.retrieval_kwargs)
    else:
        doc_filters = infer_doc_filters(state.query, companies, state.retrieval_kwargs)

    doc_query = maybe_expand_query(
        state.query,
        companies[0] if companies and not all_companies_mode and not sector_mode else None,
        hyde=runtime.settings.hyde,
        llm_client=runtime.llm_client,
        model=runtime.settings.llm_model,
    )

    doc_temporal = extract_doc_temporal_filters(state.query)
    simfin_temporal = extract_simfin_temporal_filters(state.query)
    simfin_tables = select_simfin_tables(state.query, all_companies_mode)

    logger.info(
        "Router: decision=%s all_companies=%s sector_mode=%s companies=%d tickers=%s "
        "doc_date_from=%s simfin_max_q=%d simfin_max_a=%d query='%s...'",
        decision, all_companies_mode, sector_mode, len(companies), tickers,
        doc_temporal["doc_date_from"],
        simfin_temporal["simfin_max_quarterly_periods"],
        simfin_temporal["simfin_max_annual_periods"],
        state.query[:50],
    )

    return {
        "route": decision,
        "companies": companies,
        "tickers": tickers,
        "doc_filters": doc_filters,
        "doc_query": doc_query,
        "all_companies_mode": all_companies_mode,
        "sector_mode": sector_mode,
        "doc_date_from": doc_temporal["doc_date_from"],
        "doc_date_to": doc_temporal["doc_date_to"],
        "simfin_date_from": simfin_temporal["simfin_date_from"],
        "simfin_max_quarterly_periods": simfin_temporal["simfin_max_quarterly_periods"],
        "simfin_max_annual_periods": simfin_temporal["simfin_max_annual_periods"],
        "simfin_tables": simfin_tables,
        "trace_events": [{
            "event_type": EventType.AGENT_START.value,
            "agent_name": "router",
            "payload": {
                "route": decision,
                "all_companies_mode": all_companies_mode,
                "sector_mode": sector_mode,
                "companies": companies,
                "tickers": tickers,
                "doc_query_expanded": doc_query != state.query,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    }
