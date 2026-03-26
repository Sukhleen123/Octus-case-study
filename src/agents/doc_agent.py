"""
Doc agent node: deterministic document retrieval with optional sparse-result retry.

The router pre-computes all filters (company, doc type, date range) so this node
does one targeted retrieval call — no LLM tool loop.

Phase 2 (LLM retry) only runs when initial results are sparse (< RETRY_THRESHOLD)
and an LLM client is available.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
MAX_FINAL = 15
RETRY_THRESHOLD = 3   # retry if initial retrieval returns fewer than this many chunks


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime | None:
    """Parse an ISO date string to datetime, returning None for empty strings."""
    return datetime.fromisoformat(s) if s else None


def _retrieve_documents(
    query: str,
    company_name: str | None = None,
    doc_source: str | None = None,
    document_type: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[dict]:
    """Single targeted retrieval with optional metadata and date filters."""
    return runtime.retriever.retrieve(
        query,
        company_name=company_name,
        doc_source=doc_source,
        document_type=document_type,
        date_from=date_from,
        date_to=date_to,
    )


def _retrieve_for_each_company(
    query: str,
    companies: list[str],
    doc_source: str | None = None,
    document_type: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[dict]:
    """Parallel per-company retrieval; returns up to 4 chunks per company."""
    all_chunks: list[dict] = []
    max_workers = min(8, max(1, len(companies)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                runtime.retriever.retrieve,
                query,
                company_name=co,
                doc_source=doc_source,
                document_type=document_type,
                date_from=date_from,
                date_to=date_to,
            ): co
            for co in companies
        }
        for future in as_completed(futures):
            all_chunks.extend(future.result()[:4])
    return all_chunks


# ── Citation builder ──────────────────────────────────────────────────────────

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
        cited_text=str(chunk.get("text", "")),
        company_name=chunk.get("company_name", ""),
    )


# ── Phase 2: LLM-guided retry for sparse results ──────────────────────────────

def _retry_with_llm(
    original_query: str,
    companies: list[str],
    doc_filters: dict,
    date_from: datetime | None,
    date_to: datetime | None,
) -> list[dict]:
    """Generate company-specific alternative queries via LLM and re-retrieve."""
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=runtime.settings.doc_agent_model,
        anthropic_api_key=runtime.settings.anthropic_api_key,
    )

    company_str = ", ".join(companies[:5]) if companies else ""
    prompt = (
        f"A document search returned no results for the query:\n'{original_query}'\n"
        + (f"Relevant companies: {company_str}\n" if company_str else "")
        + "\nGenerate 3 shorter, more direct search queries to find relevant earnings "
        "transcripts or SEC filings. Incorporate specific company names where relevant. "
        "Return only the queries, one per line, no numbering."
    )
    response = llm.invoke(prompt)
    alt_queries = [
        q.strip() for q in response.content.strip().split("\n")
        if q.strip() and not q.strip()[0].isdigit()
    ][:3]

    all_chunks: list[dict] = []
    seen_ids: set[str] = set()

    for alt_query in alt_queries:
        new_chunks = _retrieve_documents(
            alt_query,
            company_name=doc_filters.get("company_name"),
            doc_source=doc_filters.get("doc_source"),
            document_type=doc_filters.get("document_type"),
            date_from=date_from,
            date_to=date_to,
        )
        for c in new_chunks:
            cid = c.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(c)

    # If still sparse and a company filter exists, broaden by dropping it
    if len(all_chunks) < RETRY_THRESHOLD and doc_filters.get("company_name"):
        for alt_query in alt_queries:
            new_chunks = _retrieve_documents(
                alt_query,
                doc_source=doc_filters.get("doc_source"),
                document_type=doc_filters.get("document_type"),
                date_from=date_from,
                date_to=date_to,
            )
            for c in new_chunks:
                cid = c.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(c)

    logger.info("DocAgent retry: %d alt queries → %d new chunks", len(alt_queries), len(all_chunks))
    return all_chunks


# ── Node ──────────────────────────────────────────────────────────────────────

def doc_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: deterministic document retrieval driven by router-computed filters.

    Uses doc_date_from / doc_date_to from state for Pinecone date filtering.
    Phase 2 (LLM retry) fires when initial results are sparse (< RETRY_THRESHOLD).

    Sets: doc_result, trace_events.
    """
    events: list[Any] = [agent_start(NAME, query=state.doc_query)]

    date_from = _parse_date(state.doc_date_from)
    date_to = _parse_date(state.doc_date_to)

    use_parallel = state.all_companies_mode or (state.sector_mode and len(state.companies) > 1)

    if use_parallel:
        tool_name = "retrieve_for_each_company"
        events.append(tool_call_start(
            NAME, tool=tool_name,
            query=state.doc_query,
            n_companies=len(state.companies),
            subagent_num=1, total_subagents=1,
        ))
        chunks = _retrieve_for_each_company(
            state.doc_query,
            state.companies,
            doc_source=state.doc_filters.get("doc_source"),
            document_type=state.doc_filters.get("document_type"),
            date_from=date_from,
            date_to=date_to,
        )
    else:
        tool_name = "retrieve_documents"
        extra = {k: v for k, v in state.doc_filters.items() if v}
        events.append(tool_call_start(
            NAME, tool=tool_name,
            query=state.doc_query,
            subagent_num=1, total_subagents=1,
            **extra,
        ))
        chunks = _retrieve_documents(
            state.doc_query,
            company_name=state.doc_filters.get("company_name"),
            doc_source=state.doc_filters.get("doc_source"),
            document_type=state.doc_filters.get("document_type"),
            date_from=date_from,
            date_to=date_to,
        )

    events.append(tool_call_end(NAME, tool=tool_name, chunk_count=len(chunks), subagent_num=1, total_subagents=1))
    events.append(retrieval_results(NAME, count=len(chunks), chunks=chunks))

    # Phase 2: retry with LLM-reformulated queries when initial retrieval is sparse.
    # Skip for parallel (all-companies / sector) mode — retrying per-company is handled
    # by the parallel fetch itself, and re-querying broadly would mix company contexts.
    if (
        not use_parallel
        and len(chunks) < RETRY_THRESHOLD
        and runtime.llm_client is not None
    ):
        retry_chunks = _retry_with_llm(
            state.query, state.companies, state.doc_filters, date_from, date_to,
        )
        if retry_chunks:
            events.append(tool_call_start(
                NAME, tool="retry_retrieval",
                n_alt_queries=3, subagent_num=1, total_subagents=1,
            ))
            seen_orig = {c.get("chunk_id") for c in chunks}
            new_chunks = [c for c in retry_chunks if c.get("chunk_id") not in seen_orig]
            chunks = chunks + new_chunks
            events.append(tool_call_end(
                NAME, tool="retry_retrieval",
                chunk_count=len(new_chunks), subagent_num=1, total_subagents=1,
            ))
            events.append(retrieval_results(NAME, count=len(chunks), chunks=chunks))

    if not chunks:
        logger.warning("DocAgent: no chunks retrieved; returning empty result")
        events.append(citations_emitted(NAME, count=0))
        events.append(agent_end(NAME, chunk_count=0))
        return {
            "doc_result": ([], [], events),
            "trace_events": [e.to_dict() for e in events],
        }

    final_chunks = chunks[:MAX_FINAL]
    citations = [build_citation(c) for c in final_chunks]
    events.append(citations_emitted(NAME, count=len(citations)))
    events.append(agent_end(NAME, chunk_count=len(final_chunks)))

    return {
        "doc_result": (final_chunks, citations, events),
        "trace_events": [e.to_dict() for e in events],
    }
