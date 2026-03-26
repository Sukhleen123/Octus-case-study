"""
Streamlit UI — Octus Financial Intelligence

Run data ingestion before starting the app:
    python -m src.ingest

Then start the app:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env into os.environ BEFORE any LangChain/LangGraph imports so that
# LANGCHAIN_TRACING_V2 and LANGSMITH_API_KEY are visible to the tracing SDK.
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage

import logging
import sys

from src.app.loader import load_context
from src.app.settings import Settings
from src.agents.state import AgentState
from src.citations.formatter import format_citations_block
from src.citations.models import OctusCitation, SimFinCitation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

st.set_page_config(
    page_title="Octus Financial Intelligence",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_app_context():
    """Load the app context once per process (cached across all reruns)."""
    return load_context(Settings())


try:
    ctx = get_app_context()
except Exception as e:
    st.error(f"Failed to load app context: {e}")
    st.info("Run `python -m src.ingest` to populate the data artifacts first.")
    st.stop()

# ── Citation rendering ─────────────────────────────────────────────────────────

def _render_citations(citations: list[dict]) -> None:
    """Render citations — SimFin as markdown tables, Octus as expandable chunks."""
    objects = []
    for d in citations:
        if "ticker" in d:
            objects.append(SimFinCitation(**{k: v for k, v in d.items() if k != "type"}))
        else:
            objects.append(OctusCitation(**{k: v for k, v in d.items() if k != "type"}))

    sorted_cites = sorted(objects, key=lambda c: c.ref_number if c.ref_number > 0 else 999)
    simfin_cites = [c for c in sorted_cites if isinstance(c, SimFinCitation)]
    octus_cites  = [c for c in sorted_cites if isinstance(c, OctusCitation)]

    if simfin_cites:
        st.markdown(format_citations_block(simfin_cites))

    if octus_cites:
        if simfin_cites:
            st.markdown("---")
        for c in octus_cites:
            ref = f"[{c.ref_number}] " if c.ref_number else ""
            company = f" · {c.company_name}" if c.company_name else ""
            label = f"{ref}{c.doc_source.upper()} · {c.document_type} · {c.document_date[:10]}{company}"
            with st.expander(label):
                st.markdown(c.cited_text)


# ── Chat UI ────────────────────────────────────────────────────────────────────

st.title("Octus Financial Intelligence")
st.caption("Ask questions about Octus transcripts, SEC filings, and SimFin financial data.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Citations"):
                _render_citations(msg["citations"])
        if msg.get("trace"):
            with st.expander("Agent trace", expanded=False):
                for event in msg["trace"]:
                    st.json(event)

_AGENT_LABELS = {
    "router":           "Router",
    "doc_agent":        "Document Agent",
    "simfin_agent":     "SimFin Agent",
    "synthesis_agent":  "Synthesis Agent",
    "synthesize":       "Synthesis Agent",
}


_DOC_RETRIEVAL_TOOLS = ("retrieve_documents", "retrieve_for_each_company")


def _render_trace_events(node_output: dict) -> list[str]:
    """Convert trace events from a node output into human-readable status lines."""
    events = node_output.get("trace_events", [])
    lines: list[str] = []

    for event in events:
        etype = event.get("event_type", "")
        agent = event.get("agent_name", "")
        payload = event.get("payload", {})
        label = _AGENT_LABELS.get(agent, agent.replace("_", " ").title())

        if etype == "agent_start":
            tickers = payload.get("tickers", [])
            if agent == "router":
                lines.append(f"**[{label}]** Routing query…")
            elif agent == "simfin_agent" and tickers:
                lines.append(f"**[{label}]** Starting financial data fetch for {', '.join(tickers)}…")
            elif agent == "doc_agent":
                lines.append(f"**[{label}]** Starting document retrieval…")
            elif agent in ("synthesize", "synthesis_agent"):
                lines.append(f"**[{label}]** Synthesizing answer…")

        elif etype == "tool_call_start" and payload.get("tool") in _DOC_RETRIEVAL_TOOLS:
            tool = payload["tool"]
            _q = str(payload.get("query", ""))
            query = (_q[:70] + "…") if len(_q) > 70 else _q
            if tool == "retrieve_for_each_company":
                n = payload.get("n_companies") or len(payload.get("companies", []))
                lines.append(f"**[{label}]** Searching: parallel search across {n} companies — \"{query}\"")
            else:
                filters = {
                    k: v for k, v in payload.items()
                    if k not in ("tool", "query") and v is not None
                }
                filter_str = ", ".join(f"{k}={v}" for k, v in filters.items()) if filters else "no filters"
                lines.append(f"**[{label}]** Searching: \"{query}\" ({filter_str})")

        elif etype == "tool_call_start" and payload.get("tool") == "duckdb":
            ticker = payload.get("ticker", "")
            lines.append(f"**[{label}]** Fetching financial statements for {ticker}…")

        elif etype == "tool_call_end" and payload.get("tool") in _DOC_RETRIEVAL_TOOLS:
            count = payload.get("chunk_count", 0)
            lines.append(f"**[{label}]** Retrieved {count} chunks")

        elif etype == "tool_call_end" and payload.get("tool") == "duckdb":
            ticker = payload.get("ticker", "")
            row_count = payload.get("row_count", 0)
            tables = payload.get("tables_fetched", [])
            tables_str = ", ".join(t.replace("_", " ") for t in tables) if tables else "no data"
            lines.append(f"**[{label}]** {ticker}: {row_count} rows ({tables_str})")

        elif etype == "citations_emitted" and agent == "doc_agent":
            count = payload.get("count", 0)
            lines.append(f"**[{label}]** Selected {count} relevant chunks for synthesis")

    return lines

if prompt := st.chat_input("Ask about financials, earnings, filings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        initial = AgentState(
            query=prompt,
            messages=[HumanMessage(content=prompt)],
        )
        accumulated: dict = {}
        answer = "Error: no result produced."
        citations: list[dict] = []
        trace: list = []

        with st.status("Analyzing your question…", expanded=True) as status:
            try:
                for update in ctx.graph.stream(
                    initial.model_dump(), stream_mode="updates"
                ):
                    for node_name, node_output in update.items():
                        accumulated.update(node_output)
                        for line in _render_trace_events(node_output):
                            st.write(line)
                status.update(label="Complete", state="complete")
            except Exception as e:
                status.update(label=f"Error: {e}", state="error")

        result = accumulated.get("synthesis_result")
        if result is not None:
            answer = result.final_answer_text
            citations = result.citations
            trace = accumulated.get("trace_events", [])

        st.markdown(answer)

        if citations:
            with st.expander("Citations"):
                _render_citations(citations)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "trace": trace,
    })
