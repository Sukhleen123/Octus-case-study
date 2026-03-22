"""
Streamlit UI — Financial Multi-Agent Intelligence System

Default workflow:
    streamlit run apps/streamlit_app.py

Architecture:
  - Settings loaded via Pydantic BaseSettings
  - ensure_ready(settings) called under st.cache_resource (idempotent + file-locked)
  - AppContext (vectorstore, retriever, orchestrator) reused across reruns
  - Settings UI with auto-rebuild on change (apply_mode="auto")
  - Chat interface shows: agent routing, tool calls, retrieved sources, citations

References:
  - st.cache_resource: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
  - Streamlit caching: https://docs.streamlit.io/develop/concepts/architecture/caching
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.app.bootstrap import ensure_ready
from src.app.logging import configure_logging
from src.app.settings import Settings
from src.citations.formatter import format_citations_block

configure_logging()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Octus Financial Intelligence",
    page_icon="📊",
    layout="wide",
)

# ── Settings + Bootstrap (cached across all reruns in this process) ────────────

@st.cache_resource
def get_app_context(settings_key: str):
    """
    Cache the AppContext so bootstrap only runs once per settings combination.

    settings_key is a deterministic hash of the settings; changing it causes
    st.cache_resource to recompute (auto-rebuild on settings change).
    """
    settings = Settings()
    return ensure_ready(settings)


def _settings_key(s: Settings) -> str:
    """Deterministic key for cache invalidation."""
    import json
    return json.dumps(s.fingerprint_subset(), sort_keys=True)


# ── Sidebar: Settings UI ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    settings = Settings()

    table_format = st.selectbox(
        "Table format", ["parquet", "csv"], index=0 if settings.table_format == "parquet" else 1
    )
    chunk_size = st.slider("Chunk size (tokens)", 128, 2048, settings.chunk_size, step=64)
    chunk_overlap = st.slider("Chunk overlap (tokens)", 0, 256, settings.chunk_overlap, step=16)
    retriever_id = st.selectbox(
        "Retriever", ["dense", "dense_mmr"], index=0 if settings.retriever_id == "dense" else 1
    )
    simfin_mode = st.selectbox(
        "SimFin mode", ["batch", "realtime"],
        index=0 if settings.simfin_mode == "batch" else 1,
    )
    mapping_mode = st.selectbox(
        "Mapping mode", ["auto_matched", "confirmed", "both"],
        index=["auto_matched", "confirmed", "both"].index(settings.mapping_mode),
    )

    # Build effective settings from UI values
    effective_settings = Settings(
        table_format=table_format,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retriever_id=retriever_id,
        simfin_mode=simfin_mode,
        mapping_mode=mapping_mode,
    )

    skey = _settings_key(effective_settings)

    if settings.apply_mode == "manual":
        # Manual mode: show "Apply changes" button
        if st.button("Apply changes", type="primary"):
            st.session_state["active_settings_key"] = skey
    else:
        # Auto mode: apply immediately on any change
        st.session_state["active_settings_key"] = skey

    active_key = st.session_state.get("active_settings_key", _settings_key(settings))
    st.caption(f"Fingerprint: `{active_key[:16]}...`")

    with st.expander("Vectorstore info"):
        st.write(f"Provider: `{settings.vectorstore_provider}`")
        st.write(f"Pinecone key set: `{bool(settings.pinecone_api_key)}`")
        st.write(f"Embedding: `{settings.embedding_provider}`")

# ── Bootstrap ──────────────────────────────────────────────────────────────────

try:
    ctx = get_app_context(active_key)
except Exception as e:
    st.error(f"Bootstrap failed: {e}")
    st.stop()

# ── Main chat area ─────────────────────────────────────────────────────────────

st.title("Octus Financial Intelligence")
st.caption(
    "Ask questions about Octus transcripts, SEC filings, and SimFin financial data."
)

# Chat history
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

# Chat input
if prompt := st.chat_input("Ask about financials, earnings, filings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ctx.orchestrator.run(prompt)
                answer = result.final_answer_text
                citations_md = format_citations_block(result.citations)
            except Exception as e:
                answer = f"Error: {e}"
                citations_md = ""
                result = None

        st.markdown(answer)

        if citations_md:
            with st.expander("Citations"):
                st.markdown(citations_md)

        trace = result.trace_events if result else []
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
