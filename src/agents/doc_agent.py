"""
Doc agent node: retrieves Octus document chunks and builds citations.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents import runtime
from src.agents.events import (
    agent_end,
    agent_start,
    citations_emitted,
    retrieval_results,
    tool_call_end,
    tool_call_start,
)
from src.agents.state import AgentState
from src.citations.models import OctusCitation

logger = logging.getLogger(__name__)

NAME = "doc_agent"


def build_citation(chunk: dict) -> OctusCitation:
    """Build an OctusCitation from a retrieved chunk dict."""
    doc_date = chunk.get("document_date", "")
    if hasattr(doc_date, "isoformat"):
        doc_date = doc_date.isoformat()
    return OctusCitation(
        document_id=chunk.get("document_id", ""),
        doc_source=chunk.get("doc_source", ""),
        document_type=chunk.get("document_type", ""),
        document_date=str(doc_date),
        chunk_id=chunk.get("chunk_id", ""),
        cited_text=str(chunk.get("text", "")).replace("\n", " "),
    )


def doc_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: retrieves document chunks via the runtime retriever.

    Sets: doc_result, trace_events.
    """
    events = []
    events.append(agent_start(NAME, query=state.doc_query))
    events.append(tool_call_start(NAME, tool="retriever", query=state.doc_query))

    chunks = runtime.retriever.retrieve(state.doc_query, **state.doc_filters)

    events.append(tool_call_end(NAME, tool="retriever", chunk_count=len(chunks)))
    events.append(retrieval_results(NAME, count=len(chunks), chunks=chunks))

    logger.info("DocAgent retrieved %d chunks for query: %s...", len(chunks), state.doc_query[:60])

    citations = [build_citation(chunk) for chunk in chunks]

    events.append(citations_emitted(NAME, count=len(citations)))
    events.append(agent_end(NAME, chunk_count=len(chunks)))

    return {
        "doc_result": (chunks, citations, events),
        "trace_events": [e.to_dict() for e in events],
    }
