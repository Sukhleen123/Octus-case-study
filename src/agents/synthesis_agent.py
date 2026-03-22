"""
Synthesis agent: merges doc + simfin results into a final answer with inline citations.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.events import TraceEvent, agent_end, agent_start, citations_emitted
from src.citations.models import OctusCitation, SimFinCitation, SynthesisResult

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """
    Merges results from DocAgent and SimFinAgent.

    In mock mode: produces a templated answer from retrieved excerpts.
    With an LLM: numbers all citations [1..N], builds a sourced context,
    instructs Claude to cite inline with [N] markers, then maps refs back
    to citation objects — only citations actually used in the answer are returned.
    """

    NAME = "synthesis_agent"

    def __init__(self, llm_client: Any = None) -> None:
        self._llm = llm_client  # None → mock mode

    def run(
        self,
        query: str,
        doc_result: tuple[list[dict], list[OctusCitation], list[TraceEvent]],
        simfin_result: tuple[list[Any], list[SimFinCitation], list[TraceEvent]],
    ) -> SynthesisResult:
        """
        Synthesize a final answer.

        Args:
            query: Original user query.
            doc_result: (chunks, octus_citations, doc_events) from DocAgent.
            simfin_result: (tables, simfin_citations, sf_events) from SimFinAgent.

        Returns:
            SynthesisResult with final_answer_text, citations[], trace_events[].
        """
        chunks, octus_cites, doc_events = doc_result
        tables, simfin_cites, sf_events = simfin_result

        all_events: list[TraceEvent] = doc_events + sf_events
        all_events.append(agent_start(self.NAME, query=query))

        if self._llm is not None:
            answer, used_refs = self._llm_answer(query, octus_cites, simfin_cites)
            # Assign inline ref numbers and keep only cited citations
            ordered = list(simfin_cites) + list(octus_cites)
            for ref, cit in enumerate(ordered, 1):
                if ref in used_refs:
                    cit.ref_number = ref
            all_citations: list[OctusCitation | SimFinCitation] = [
                c for c in ordered if c.ref_number > 0
            ]
        else:
            answer = self._mock_answer(query, chunks, tables)
            all_citations = list(simfin_cites) + list(octus_cites)

        all_events.append(citations_emitted(self.NAME, count=len(all_citations)))
        all_events.append(agent_end(self.NAME))

        return SynthesisResult(
            final_answer_text=answer,
            citations=all_citations,
            trace_events=[e.to_dict() for e in all_events],
        )

    def _mock_answer(self, query: str, chunks: list[dict], tables: list) -> str:
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

    def _llm_answer(
        self,
        query: str,
        octus_cites: list[OctusCitation],
        simfin_cites: list[SimFinCitation],
    ) -> tuple[str, set[int]]:
        """
        Call the configured LLM with numbered citations as context.

        SimFin citations come first (structured data), then Octus doc excerpts.
        Claude is instructed to place [N] markers inline after each supported claim.

        Returns:
            (answer_text, used_refs) where used_refs is the set of [N] numbers
            that appear in the answer.
        """
        # Number citations: simfin first, then octus
        ordered = list(simfin_cites) + list(octus_cites)

        context_lines: list[str] = []
        for ref, cit in enumerate(ordered, 1):
            if isinstance(cit, SimFinCitation):
                value_str = (
                    f" = {cit.metric_value} ({cit.metric_unit})"
                    if cit.metric_value
                    else ""
                )
                context_lines.append(
                    f"[{ref}] [SimFin: {cit.ticker} {cit.fiscal_period} "
                    f"FY{cit.fiscal_year}] {cit.statement_type} — "
                    f"{cit.metric_name}{value_str}"
                )
            else:
                context_lines.append(
                    f"[{ref}] [{cit.doc_source.upper()} / {cit.document_type} "
                    f"({cit.document_date[:10]})] {cit.cited_text}"
                )

        if not context_lines:
            context_lines.append("(No sources available)")

        context = "\n".join(context_lines)
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
            f"Question: {query}\n\n"
            f"Sources:\n{context}\n\n"
            "Answer:"
        )

        # Anthropic Claude
        if hasattr(self._llm, "messages"):
            response = self._llm.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_text = response.content[0].text
        else:
            answer_text = f"[LLM] {prompt[:100]}..."

        used_refs = {int(m) for m in re.findall(r'\[(\d+)\]', answer_text)}
        return answer_text, used_refs
