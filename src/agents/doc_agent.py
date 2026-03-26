"""
Doc agent node: LLM-driven document retrieval with tool-calling and structured chunk selection.

Phase 1 — Tool loop: LLM calls retrieval tools (retrieve_documents,
retrieve_for_each_company), loops up to MAX_ITER rounds, stopping when
satisfied. Chunks are deduplicated across rounds by chunk_id.

Phase 2 — Selection: LLM selects the top MAX_FINAL chunks most relevant
to the user's query using structured output (DocAgentSelection).

Fallback: if runtime.llm_client is None, falls back to a single
retriever.retrieve() call (no LLM decisions).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

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
MAX_ITER = 3
MAX_FINAL = 15


# ── Tool input schemas ────────────────────────────────────────────────────────

class RetrieveDocumentsInput(BaseModel):
    query: str
    company_name: str | None = Field(default=None, description="Filter to a specific company name")
    doc_source: str | None = Field(default=None, description="'transcript' or 'sec_filing'")
    document_type: str | None = Field(default=None, description="'10-K', '10-Q', or 'Transcript'")


class RetrieveForEachCompanyInput(BaseModel):
    query: str = Field(description="Search query to run for each company")
    companies: list[str] = Field(description="List of company names to retrieve for in parallel")
    doc_source: str | None = Field(default=None, description="'transcript' or 'sec_filing'")
    document_type: str | None = Field(default=None, description="'10-K', '10-Q', or 'Transcript'")


# ── Tool implementations ──────────────────────────────────────────────────────

def _retrieve_documents(
    query: str,
    company_name: str | None = None,
    doc_source: str | None = None,
    document_type: str | None = None,
) -> list[dict]:
    """Single targeted retrieval with optional metadata filters."""
    return runtime.retriever.retrieve(
        query,
        company_name=company_name,
        doc_source=doc_source,
        document_type=document_type,
    )


def _retrieve_for_each_company(
    query: str,
    companies: list[str],
    doc_source: str | None = None,
    document_type: str | None = None,
) -> list[dict]:
    """Parallel per-company retrieval; returns up to 2 chunks per company."""
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
            ): co
            for co in companies
        }
        for future in as_completed(futures):
            all_chunks.extend(future.result()[:2])
    return all_chunks


retrieve_documents_tool = StructuredTool.from_function(
    func=_retrieve_documents,
    name="retrieve_documents",
    description=(
        "Retrieve document chunks matching a search query. "
        "Use for targeted retrieval for a specific query, optionally filtered to a single company, "
        "document source (transcript/sec_filing), or document type (10-K/10-Q/Transcript)."
    ),
    args_schema=RetrieveDocumentsInput,
)

retrieve_for_each_company_tool = StructuredTool.from_function(
    func=_retrieve_for_each_company,
    name="retrieve_for_each_company",
    description=(
        "Retrieve document chunks for MULTIPLE companies in parallel. "
        "Use when you need the same information (e.g. CEO name, guidance, risk factors) "
        "across a large list of companies. Specify the query and the full list of company names."
    ),
    args_schema=RetrieveForEachCompanyInput,
)

TOOLS = [retrieve_documents_tool, retrieve_for_each_company_tool]


# ── Structured selection output ───────────────────────────────────────────────

class SelectedChunk(BaseModel):
    chunk_id: str
    text: str
    doc_source: str
    document_type: str
    document_date: str
    company_name: str
    score: float


class DocAgentSelection(BaseModel):
    selected_chunks: list[SelectedChunk]
    rationale: str


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _execute_tool(tool_call: dict) -> list[dict]:
    name = tool_call["name"]
    args = tool_call.get("args", {})
    if name == "retrieve_documents":
        return _retrieve_documents(**args)
    if name == "retrieve_for_each_company":
        return _retrieve_for_each_company(**args)
    logger.warning("DocAgent: unknown tool '%s'", name)
    return []


# ── Node ──────────────────────────────────────────────────────────────────────

def doc_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: LLM-driven retrieval with tool loop and structured chunk selection.

    Phase 1: LLM calls retrieve_documents or retrieve_for_each_company tools,
    loops up to MAX_ITER rounds, deduplicating chunks across rounds by chunk_id.

    Phase 2: LLM selects the top MAX_FINAL chunks most relevant to the query
    using structured output (DocAgentSelection).

    Fallback: if runtime.llm_client is None, single retriever.retrieve() call.

    Sets: doc_result, trace_events.
    """
    events: list[Any] = []
    events.append(agent_start(NAME, query=state.doc_query))

    # ── Fallback: no LLM ─────────────────────────────────────────────────────
    if runtime.llm_client is None:
        events.append(tool_call_start(NAME, tool="retriever", query=state.doc_query))
        chunks = runtime.retriever.retrieve(state.doc_query, **state.doc_filters)
        events.append(tool_call_end(NAME, tool="retriever", chunk_count=len(chunks)))
        events.append(retrieval_results(NAME, count=len(chunks), chunks=chunks))
        citations = [build_citation(c) for c in chunks]
        events.append(citations_emitted(NAME, count=len(citations)))
        events.append(agent_end(NAME, chunk_count=len(chunks)))
        return {
            "doc_result": (chunks, citations, events),
            "trace_events": [e.to_dict() for e in events],
        }

    # ── Phase 1: Tool-calling loop ────────────────────────────────────────────
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=runtime.settings.doc_agent_model,
        anthropic_api_key=runtime.settings.anthropic_api_key,
    ).bind_tools(TOOLS)

    companies_context = (
        f"Available companies ({len(state.companies)}): {', '.join(state.companies)}"
        if state.companies
        else "No specific companies identified."
    )
    system_content = (
        "You are a document retrieval agent. Use the available tools to retrieve "
        "document chunks that will help answer the user's query. Retrieve from "
        "multiple angles if needed (e.g. transcripts AND SEC filings). "
        "Stop calling tools when you have gathered enough evidence.\n\n"
        + companies_context
    )

    messages: list[Any] = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Query: {state.query}"),
    ]

    all_chunks: list[dict] = []
    seen_ids: set[str] = set()
    round_num = 0

    for round_num in range(MAX_ITER):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_messages: list[ToolMessage] = []
        for tc in response.tool_calls:
            events.append(tool_call_start(NAME, tool=tc["name"], **tc["args"]))
            new_chunks = _execute_tool(tc)
            deduped = [c for c in new_chunks if c.get("chunk_id") not in seen_ids]
            for c in deduped:
                if c.get("chunk_id"):
                    seen_ids.add(c["chunk_id"])
            all_chunks.extend(deduped)
            events.append(tool_call_end(NAME, tool=tc["name"], chunk_count=len(deduped)))
            tool_messages.append(ToolMessage(
                content=f"Retrieved {len(deduped)} new chunks (deduplicated).",
                tool_call_id=tc["id"],
            ))

        messages.extend(tool_messages)
        events.append(retrieval_results(NAME, count=len(all_chunks), chunks=all_chunks))

    logger.info(
        "DocAgent Phase 1: %d rounds, %d total unique chunks",
        round_num + 1, len(all_chunks),
    )

    # ── Phase 2: Query-aware chunk selection ──────────────────────────────────
    if not all_chunks:
        logger.warning("DocAgent: no chunks retrieved; returning empty result")
        events.append(agent_end(NAME, chunk_count=0))
        return {
            "doc_result": ([], [], events),
            "trace_events": [e.to_dict() for e in events],
        }

    selection_llm = ChatAnthropic(
        model=runtime.settings.doc_agent_model,
        anthropic_api_key=runtime.settings.anthropic_api_key,
    ).with_structured_output(DocAgentSelection)

    chunk_summaries = "\n".join(
        f"[{c.get('chunk_id', '?')}] "
        f"({c.get('company_name', '?')}, {c.get('document_type', '?')}, "
        f"{str(c.get('document_date', '?'))[:10]}, "
        f"score={c.get('_score', 0.0):.3f}): "
        f"{str(c.get('text', ''))[:200]}"
        for c in all_chunks
    )
    selection_prompt = (
        f"Query: {state.query}\n\n"
        f"Retrieved chunks:\n{chunk_summaries}\n\n"
        f"Select up to {MAX_FINAL} chunks most relevant to answering the query. "
        f"Include only chunks that provide substantive evidence. "
        f"Populate each SelectedChunk with its full metadata and text from the list above."
    )
    selection: DocAgentSelection = selection_llm.invoke(selection_prompt)

    # Hydrate final chunks from all_chunks by chunk_id
    chunk_by_id = {c["chunk_id"]: c for c in all_chunks if c.get("chunk_id")}
    final_chunks = [
        chunk_by_id[sc.chunk_id]
        for sc in selection.selected_chunks
        if sc.chunk_id in chunk_by_id
    ]

    logger.info(
        "DocAgent Phase 2: %d selected from %d. Rationale: %s",
        len(final_chunks), len(all_chunks), selection.rationale[:120],
    )

    citations = [build_citation(c) for c in final_chunks]
    events.append(citations_emitted(NAME, count=len(citations)))
    events.append(agent_end(NAME, chunk_count=len(final_chunks)))

    return {
        "doc_result": (final_chunks, citations, events),
        "trace_events": [e.to_dict() for e in events],
    }
