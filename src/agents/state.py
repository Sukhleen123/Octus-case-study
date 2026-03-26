"""
AgentState and SynthesisOutput Pydantic models for the LangGraph pipeline.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class SynthesisOutput(BaseModel):
    """Structured output produced by the synthesis node."""

    final_answer_text: str
    citations: list[dict]      # serialized OctusCitation / SimFinCitation dicts
    trace_events: list[dict]


class AgentState(BaseModel):
    """Full pipeline state threaded through every graph node."""

    # ── Inputs ───────────────────────────────────────────────────────────────
    query: str = ""
    retrieval_kwargs: dict = Field(default_factory=dict)

    # ── Set by router_node ────────────────────────────────────────────────────
    route: str = ""
    companies: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    doc_filters: dict = Field(default_factory=dict)
    doc_query: str = ""
    all_companies_mode: bool = False

    # ── Set by doc_agent_node ─────────────────────────────────────────────────
    doc_result: tuple = Field(default=([], [], []))

    # ── Set by simfin_agent_node ──────────────────────────────────────────────
    simfin_result: tuple = Field(default=([], [], []))

    # ── Set by synthesize_node ────────────────────────────────────────────────
    synthesis_result: SynthesisOutput | None = None

    # ── Accumulated across all nodes via operator.add reducer ─────────────────
    trace_events: Annotated[list[dict], operator.add] = Field(default_factory=list)

    # ── Conversation history via add_messages reducer ──────────────────────────
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
