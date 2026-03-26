"""
Router node: decides which agents to invoke and prepares shared state.

Performs keyword-based routing, company/ticker extraction, doc filter
inference, and optional HyDE query expansion before any retrieval happens.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from src.agents import runtime
from src.agents.consts import ALL_COMPANIES_TRIGGERS, DOC_TYPE_KEYWORDS, HYDE_TRIGGERS, OCTUS_KEYWORDS, SEC_KEYWORDS, SIMFIN_KEYWORDS
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

    if all_companies_mode and runtime.company_map is not None:
        import pandas as pd
        df = runtime.company_map
        companies = df["company_name"].dropna().unique().tolist() if isinstance(df, pd.DataFrame) else []
    else:
        companies = extract_companies(state.query, runtime.company_map)

    tickers = resolve_tickers(companies, runtime.company_map)
    decision = route_decision(state.query)

    # All-companies queries with financial data requests always hit both agents
    if all_companies_mode and tickers:
        decision = "both"

    # In all-companies mode, omit company_name from doc_filters — doc_agent
    # applies per-company filters itself via parallel retrieval
    if all_companies_mode:
        doc_filters = infer_doc_filters(state.query, [], state.retrieval_kwargs)
    else:
        doc_filters = infer_doc_filters(state.query, companies, state.retrieval_kwargs)

    doc_query = maybe_expand_query(
        state.query,
        companies[0] if companies and not all_companies_mode else None,
        hyde=runtime.settings.hyde,
        llm_client=runtime.llm_client,
        model=runtime.settings.llm_model,
    )

    logger.info(
        "Router: decision=%s all_companies=%s companies=%d tickers=%s query='%s...'",
        decision, all_companies_mode, len(companies), tickers, state.query[:50],
    )

    return {
        "route": decision,
        "companies": companies,
        "tickers": tickers,
        "doc_filters": doc_filters,
        "doc_query": doc_query,
        "all_companies_mode": all_companies_mode,
        "trace_events": [{
            "event_type": EventType.AGENT_END.value,
            "agent_name": "router",
            "payload": {
                "route": decision,
                "all_companies_mode": all_companies_mode,
                "companies": companies,
                "tickers": tickers,
                "doc_query_expanded": doc_query != state.query,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    }
