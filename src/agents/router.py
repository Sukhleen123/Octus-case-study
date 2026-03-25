"""
Router node: decides which agents to invoke and prepares shared state.

Performs keyword-based routing, company/ticker extraction, doc filter
inference, and optional HyDE query expansion before any retrieval happens.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from src.agents import runtime
from src.agents.consts import OCTUS_KEYWORDS, SIMFIN_KEYWORDS
from src.agents.events import EventType
from src.agents.state import AgentState
from src.agents.utils import (
    extract_companies,
    infer_doc_filters,
    maybe_expand_query,
    resolve_tickers,
)

logger = logging.getLogger(__name__)

RouteDecision = Literal["octus", "simfin", "both"]


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

    Sets: route, companies, tickers, doc_filters, doc_query, trace_events.
    """
    decision = route_decision(state.query)
    companies = extract_companies(state.query, runtime.company_map)
    tickers = resolve_tickers(companies, runtime.company_map)
    doc_filters = infer_doc_filters(state.query, companies, state.retrieval_kwargs)
    doc_query = maybe_expand_query(
        state.query,
        companies[0] if companies else None,
        hyde=runtime.settings.hyde,
        llm_client=runtime.llm_client,
        model=runtime.settings.llm_model,
    )

    logger.info(
        "Router: decision=%s companies=%s tickers=%s query='%s...'",
        decision, companies, tickers, state.query[:50],
    )

    return {
        "route": decision,
        "companies": companies,
        "tickers": tickers,
        "doc_filters": doc_filters,
        "doc_query": doc_query,
        "trace_events": [{
            "event_type": EventType.AGENT_END.value,
            "agent_name": "router",
            "payload": {
                "route": decision,
                "companies": companies,
                "tickers": tickers,
                "doc_query_expanded": doc_query != state.query,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    }
