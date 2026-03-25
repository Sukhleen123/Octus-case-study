"""
Application context loader.

Connects to pre-built artifacts (Pinecone namespaces + DuckDB), initializes
the agent runtime singletons, compiles the LangGraph, and returns an AppContext
ready for the Streamlit app.

Run `python -m src.ingest` first to populate the artifacts.

Usage:
    from src.app.loader import load_context
    from src.app.settings import Settings

    ctx = load_context(Settings())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.app.context import AppContext
from src.app.settings import Settings

logger = logging.getLogger(__name__)


def load_context(settings: Settings) -> AppContext:
    """
    Connect to pre-built Pinecone namespaces and DuckDB, initialize the agent
    runtime, and compile the LangGraph.

    Does NOT rebuild any artifacts — run `python -m src.ingest` for that.
    """
    from src.agents import runtime
    from src.agents.graph import build_graph
    from src.embeddings.embedder import get_embedder
    from src.retrieval.multi_store import MultiStoreRetriever
    from src.vectorstore.metadata_store import MetadataStore
    from src.vectorstore.pinecone_store import PineconeStore

    embedder = get_embedder(settings)
    cache_dir = Path(settings.cache_dir) / "pinecone"

    transcript_store = PineconeStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_transcripts_namespace,
        dim=settings.embedding_dim,
        text_store=MetadataStore(cache_dir / "transcripts_text_store.sqlite"),
    )
    sec_store = PineconeStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_sec_namespace,
        dim=settings.embedding_dim,
        text_store=MetadataStore(cache_dir / "sec_filings_text_store.sqlite"),
    )

    transcript_retriever = _make_retriever(settings, transcript_store, embedder)
    sec_retriever = _make_retriever(settings, sec_store, embedder)

    retriever = MultiStoreRetriever(
        transcript_retriever=transcript_retriever,
        sec_retriever=sec_retriever,
        top_k=settings.top_k,
    )

    company_map = _load_company_map(settings)
    llm_client = _make_llm_client(settings)

    # Initialize runtime singletons so node functions can import them directly
    runtime.init(retriever, llm_client, company_map)

    graph = build_graph()

    logger.info(
        "App context loaded. Transcripts: %d vectors, SEC filings: %d vectors",
        transcript_store.count(),
        sec_store.count(),
    )

    return AppContext(retriever=retriever, graph=graph)


def _make_retriever(settings: Settings, store: Any, embedder: Any) -> Any:
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.dense_mmr import DenseMMRRetriever

    if settings.retriever_id == "dense_mmr":
        return DenseMMRRetriever(
            vectorstore=store,
            embedder=embedder,
            top_k=settings.top_k,
            mmr_lambda=settings.mmr_lambda,
        )
    return DenseRetriever(vectorstore=store, embedder=embedder, top_k=settings.top_k)


def _load_company_map(settings: Settings) -> Any:
    from src.simfin.storage import read_table

    map_path = Path(settings.simfin_processed_dir) / f"company_map.{settings.table_format}"
    if map_path.exists():
        logger.info("Loading company map from %s", map_path)
        return read_table(map_path)
    logger.warning("Company map not found at %s — company resolution disabled", map_path)
    return None


def _make_llm_client(settings: Settings) -> Any:
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return None
