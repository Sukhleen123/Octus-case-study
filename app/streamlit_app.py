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

import streamlit as st
from langchain_core.messages import HumanMessage

import logging
import sys

from src.app.loader import load_context
from src.app.settings import Settings
from src.agents.state import AgentState
from src.citations.formatter import format_citations_from_dicts

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
                st.markdown(msg["citations"])
        if msg.get("trace"):
            with st.expander("Agent trace", expanded=False):
                for event in msg["trace"]:
                    st.json(event)

_NODE_LABELS = {
    "router":       lambda _: "(Routing Node) Routing query…",
    "doc_agent":    lambda s: (
        "(Document Agent) Retrieving documents for "
        + (", ".join(s.get("companies") or ["your query"]) + "…")
    ),
    "simfin_agent": lambda s: (
        "(Simfin Agent) Fetching financial data"
        + (f" for {', '.join(s['tickers'])}" if s.get("tickers") else "")
        + "…"
    ),
    "synthesize":   lambda _: "(Synthesis Agent) Synthesizing answer…",
}

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
        citations_md = ""
        trace: list = []

        with st.status("Analyzing your question…", expanded=True) as status:
            try:
                for update in ctx.graph.stream(
                    initial.model_dump(), stream_mode="updates"
                ):
                    for node_name, node_output in update.items():
                        accumulated.update(node_output)
                        label_fn = _NODE_LABELS.get(node_name)
                        if label_fn:
                            st.write(label_fn(accumulated))
                status.update(label="Complete", state="complete")
            except Exception as e:
                status.update(label=f"Error: {e}", state="error")

        result = accumulated.get("synthesis_result")
        if result is not None:
            answer = result.final_answer_text
            citations_md = format_citations_from_dicts(result.citations)
            trace = accumulated.get("trace_events", [])

        st.markdown(answer)

        if citations_md:
            with st.expander("Citations"):
                st.markdown(citations_md)

        if trace:
            with st.expander("Agent trace", expanded=False):
                for event in trace:
                    st.json(event)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations_md,
        "trace": trace,
    })
