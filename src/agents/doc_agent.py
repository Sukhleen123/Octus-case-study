"""
Document agent: retrieves Octus chunks and returns excerpts + citations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.agents.events import (
    TraceEvent,
    agent_end,
    agent_start,
    citations_emitted,
    retrieval_results,
    tool_call_end,
    tool_call_start,
)
from src.citations.models import OctusCitation

logger = logging.getLogger(__name__)


class DocAgent:
    """
    Retrieves relevant Octus document chunks for a query.

    Returns excerpts (text snippets) and structured OctusCitation objects.
    """

    NAME = "doc_agent"

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    def run(
        self,
        query: str,
        company_name: str | None = None,
        doc_source: str | None = None,
        document_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], list[OctusCitation], list[TraceEvent]]:
        """
        Retrieve relevant chunks.

        Returns:
            (excerpts, citations, trace_events)
            - excerpts: list of chunk dicts (text + metadata)
            - citations: OctusCitation objects for each retrieved chunk
            - trace_events: trace events for UI rendering
        """
        events: list[TraceEvent] = []
        events.append(agent_start(self.NAME, query=query))
        events.append(tool_call_start(self.NAME, tool="retriever", query=query))

        chunks = self._retriever.retrieve(
            query=query,
            company_name=company_name,
            doc_source=doc_source,
            document_type=document_type,
            date_from=date_from,
            date_to=date_to,
        )

        events.append(tool_call_end(self.NAME, tool="retriever", chunk_count=len(chunks)))
        events.append(retrieval_results(self.NAME, count=len(chunks)))
        logger.info("DocAgent retrieved %d chunks for query: %s...", len(chunks), query[:60])

        citations = []
        for chunk in chunks:
            doc_date = chunk.get("document_date", "")
            if hasattr(doc_date, "isoformat"):
                doc_date = doc_date.isoformat()
            citations.append(
                OctusCitation(
                    document_id=chunk.get("document_id", ""),
                    doc_source=chunk.get("doc_source", ""),
                    document_type=chunk.get("document_type", ""),
                    document_date=str(doc_date),
                    chunk_id=chunk.get("chunk_id", ""),
                    cited_text=str(chunk.get("text", "")).replace("\n", " "),
                )
            )

        events.append(citations_emitted(self.NAME, count=len(citations)))
        events.append(agent_end(self.NAME, chunk_count=len(chunks)))

        return chunks, citations, events
