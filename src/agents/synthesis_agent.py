"""
Synthesis node: merges doc + simfin results into a final answer with inline citations.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage

from src.agents import runtime
from src.agents.events import agent_end, agent_start, citations_emitted
from src.agents.state import AgentState, SynthesisOutput
from src.citations.models import OctusCitation, SimFinCitation

logger = logging.getLogger(__name__)

NAME = "synthesis_agent"


def build_context(
    octus_cites: list[OctusCitation],
    simfin_cites: list[SimFinCitation],
) -> tuple[list[OctusCitation | SimFinCitation], str]:
    """Number all citations and build a context string for the LLM prompt."""
    ordered: list[OctusCitation | SimFinCitation] = list(simfin_cites) + list(octus_cites)
    lines: list[str] = []

    for ref, cit in enumerate(ordered, 1):
        if isinstance(cit, SimFinCitation):
            value_str = (
                f" = {cit.metric_value} ({cit.metric_unit})" if cit.metric_value else ""
            )
            lines.append(
                f"[{ref}] [SimFin: {cit.ticker} {cit.fiscal_period} "
                f"FY{cit.fiscal_year}] {cit.statement_type} — {cit.metric_name}{value_str}"
            )
        else:
            lines.append(
                f"[{ref}] [{cit.doc_source.upper()} / {cit.document_type} "
                f"({cit.document_date[:10]})] {cit.cited_text}"
            )

    return ordered, "\n".join(lines) if lines else "(No sources available)"


def mock_answer(query: str, chunks: list[dict], tables: list) -> str:
    """Produce a templated answer when no LLM is configured."""
    parts = [f"**Query:** {query}\n"]

    if chunks:
        parts.append(f"**Retrieved {len(chunks)} document chunk(s):**")
        for i, c in enumerate(chunks[:3], 1):
            text_preview = c.get("text", "")[:200].replace("\n", " ")
            parts.append(f"  {i}. [{c.get('doc_source','')}] {text_preview}...")

    if tables:
        parts.append(f"\n**Retrieved {len(tables)} financial table(s) from SimFin.**")

    if not chunks and not tables:
        parts.append(
            "_No results found. Try broadening your query or checking data availability._"
        )

    return "\n".join(parts)


def llm_answer(
    query: str,
    ordered: list[OctusCitation | SimFinCitation],
    context: str,
    llm_client: Any,
    model: str,
    all_companies: list[str] | None = None,
) -> tuple[str, set[int]]:
    """Call Claude with numbered citations; return (answer_text, used_ref_numbers)."""
    companies_note = ""
    if all_companies:
        companies_note = (
            "IMPORTANT: The following companies are in the database. "
            "For any company that has no sources listed below, include it in your answer "
            "with a note that no data is available for that company:\n"
            + ", ".join(all_companies) + "\n\n"
        )

    prompt = (
        "Answer the following question using only the numbered sources below.\n"
        "Rules for citing evidence:\n"
        "1. For financial data (SimFin sources): state the metric and value "
        "directly in your sentence, then append [N]. "
        "Example: 'Revenue was 14,588,000 USD thousands in Q3 2024 [1].'\n"
        "2. For document excerpts (TRANSCRIPT / SEC_FILING sources): quote the "
        "relevant passage verbatim using quotation marks, then append [N]. "
        "Example: '...management noted \"strong leisure demand\" [3].'\n"
        "3. If a quoted passage is longer than ~40 words, place it as its own "
        "paragraph formatted as a markdown block quote (lines starting with '> '), "
        "with the [N] reference at the end of the block.\n"
        "4. If multiple sources support one claim, cite together: [1][2].\n\n"
        + companies_note
        + f"Question: {query}\n\n"
        f"Sources:\n{context}\n\n"
        "Answer:"
    )

    max_tokens = 4000 if all_companies else 2000
    if hasattr(llm_client, "messages"):
        response = llm_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        answer_text = response.content[0].text
    else:
        answer_text = f"[LLM unavailable] {prompt[:100]}..."

    used_refs = {int(m) for m in re.findall(r'\[(\d+)\]', answer_text)}
    return answer_text, used_refs


def synthesize_node(state: AgentState) -> dict:
    """
    LangGraph node: synthesizes a final answer from doc and simfin results.

    Sets: synthesis_result, trace_events, messages.
    """
    chunks, octus_cites, doc_events = state.doc_result
    tables, simfin_cites, sf_events = state.simfin_result

    prior_events = list(doc_events) + list(sf_events)
    synth_start = agent_start(NAME, query=state.query)
    all_events = prior_events + [synth_start]

    if runtime.llm_client is not None:
        ordered, context = build_context(octus_cites, simfin_cites)
        all_companies = state.companies if state.all_companies_mode else None
        answer, used_refs = llm_answer(
            state.query, ordered, context, runtime.llm_client, runtime.settings.llm_model,
            all_companies=all_companies,
        )
        for ref, cit in enumerate(ordered, 1):
            if ref in used_refs:
                cit.ref_number = ref
        all_citations: list[OctusCitation | SimFinCitation] = [
            c for c in ordered if c.ref_number > 0
        ]
    else:
        answer = mock_answer(state.query, chunks, tables)
        all_citations = list(simfin_cites) + list(octus_cites)

    synth_events = [
        synth_start,
        citations_emitted(NAME, count=len(all_citations)),
        agent_end(NAME),
    ]
    all_events.extend(synth_events)

    output = SynthesisOutput(
        final_answer_text=answer,
        citations=[c.to_dict() for c in all_citations],
        trace_events=[e.to_dict() for e in all_events],
    )

    return {
        "synthesis_result": output,
        # Only emit synthesis-specific events — doc/simfin events were already
        # rendered when those nodes completed; re-emitting them here would duplicate
        # every trace line in the UI.
        "trace_events": [e.to_dict() for e in synth_events],
        "messages": [AIMessage(content=answer)],
    }
