"""
Core eval pipeline: assembles components per experiment config and runs queries.

Isolated from the main bootstrap pipeline — uses separate FAISS indexes
under data/eval/indexes/ to avoid corrupting production artifacts.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.settings import Settings
from src.eval.models import EvalQuestion, ExperimentConfig, ExperimentResult

logger = logging.getLogger(__name__)

_EVAL_DIR = Path("data/eval")
_CHUNKS_CACHE_DIR = _EVAL_DIR / "chunks"
_INDEX_CACHE_DIR = _EVAL_DIR / "indexes"

# Module-level embedder cache: avoids reloading the same model (and the HuggingFace
# network checks that come with it) for every experiment that shares an embedding model.
_EMBEDDER_CACHE: dict[str, Any] = {}


def _cache_hash(config: ExperimentConfig) -> str:
    """Compute a hash for chunk/index caching."""
    key = f"{config.chunker}:{config.chunk_size}:{config.chunk_overlap}:{config.embedding_model}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ── Chunking ──────────────────────────────────────────────────────────────────

def _get_or_build_chunks(
    config: ExperimentConfig,
    docs_df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Build chunks for the experiment config, with caching."""
    from src.simfin.storage import read_table, write_table

    cache_key = f"chunks_{config.chunker}_{config.chunk_size}_{config.chunk_overlap}"
    cache_path = _CHUNKS_CACHE_DIR / f"{cache_key}.parquet"

    if cache_path.exists():
        logger.info("Using cached chunks: %s", cache_path)
        return read_table(cache_path)

    _CHUNKS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chunker = _make_chunker(config)

    logger.info("Chunking %d docs with %s (size=%d, overlap=%d)...",
                len(docs_df), config.chunker, config.chunk_size, config.chunk_overlap)

    all_chunks = []
    for _, row in docs_df.iterrows():
        doc = row.to_dict()
        chunks = chunker.chunk(doc)
        all_chunks.extend([c.to_dict() for c in chunks])

    chunks_df = pd.DataFrame(all_chunks)
    write_table(chunks_df, cache_path, "parquet")
    logger.info("Cached %d chunks to %s", len(chunks_df), cache_path)
    return chunks_df


def _make_chunker(config: ExperimentConfig) -> Any:
    """Instantiate the configured chunker."""
    if config.chunker == "heading":
        from src.chunking.heading_chunker import HeadingChunker
        return HeadingChunker()
    elif config.chunker == "token":
        from src.chunking.token_chunker import TokenChunker
        return TokenChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    elif config.chunker == "sec_section":
        from src.chunking.sec_section_chunker import SECSectionChunker
        return SECSectionChunker()
    elif config.chunker == "recursive":
        from src.chunking.recursive_chunker import RecursiveChunker
        return RecursiveChunker(target_size=config.chunk_size, overlap=config.chunk_overlap)
    elif config.chunker == "hybrid":
        from src.chunking.hybrid_chunker import HybridChunker
        return HybridChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    else:
        raise ValueError(f"Unknown chunker: {config.chunker}")


# ── Embedding + Indexing ──────────────────────────────────────────────────────

def _get_or_build_index(
    config: ExperimentConfig,
    chunks_df: pd.DataFrame,
    embedder: Any,
) -> Any:
    """Build a FAISS index for the experiment, with caching."""
    from src.vectorstore.faiss_store import FAISSStore

    idx_hash = _cache_hash(config)
    idx_dir = _INDEX_CACHE_DIR / idx_hash
    # FAISSStore stores its index at {cache_dir}/faiss/index.faiss (it appends "faiss/" itself)
    faiss_path = idx_dir / "faiss" / "index.faiss"

    if faiss_path.exists():
        logger.info("Using cached FAISS index: %s", idx_dir)
        store = FAISSStore(cache_dir=str(idx_dir), dim=embedder.dim)
        if store.count() > 0:
            return store
        logger.info("Cached index empty, rebuilding...")

    idx_dir.mkdir(parents=True, exist_ok=True)
    store = FAISSStore(cache_dir=str(idx_dir), dim=embedder.dim)

    texts = chunks_df["text"].fillna("").tolist()
    metadatas = chunks_df.to_dict(orient="records")

    # Embed all texts in one shot — lets sentence-transformers optimise its own
    # internal batching and shows a single progress bar instead of silent batches.
    device = getattr(getattr(embedder, "_model", None), "device", "unknown")
    logger.info("Embedding %d chunks on %s...", len(texts), device)
    all_vectors = embedder.embed_bulk(texts)

    # Upsert everything in one call so FAISSStore only writes the index file once.
    store.upsert(all_vectors, metadatas)

    logger.info("Built FAISS index with %d vectors at %s", store.count(), idx_dir)
    return store


def _make_embedder(config: ExperimentConfig) -> Any:
    """Instantiate the configured embedder, reusing a cached instance per model name."""
    from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
    if config.embedding_model not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[config.embedding_model] = SentenceTransformerEmbedder(
            model_name=config.embedding_model
        )
    return _EMBEDDER_CACHE[config.embedding_model]


def _make_retriever(
    config: ExperimentConfig,
    vectorstore: Any,
    embedder: Any,
    chunks_df: pd.DataFrame,
) -> Any:
    """Instantiate the configured retriever."""
    if config.retriever == "dense":
        from src.retrieval.dense import DenseRetriever
        return DenseRetriever(vectorstore=vectorstore, embedder=embedder, top_k=config.top_k)
    elif config.retriever == "dense_mmr":
        from src.retrieval.dense_mmr import DenseMMRRetriever
        return DenseMMRRetriever(
            vectorstore=vectorstore, embedder=embedder, top_k=config.top_k, mmr_lambda=0.5,
        )
    elif config.retriever == "bm25":
        from src.retrieval.bm25 import BM25Retriever
        chunks = chunks_df.to_dict(orient="records")
        return BM25Retriever(chunks=chunks, top_k=config.top_k)
    elif config.retriever == "hybrid_dense_bm25":
        from src.retrieval.bm25 import BM25Retriever
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.hybrid import HybridRetriever
        chunks = chunks_df.to_dict(orient="records")
        dense = DenseRetriever(vectorstore=vectorstore, embedder=embedder, top_k=config.top_k)
        bm25 = BM25Retriever(chunks=chunks, top_k=config.top_k)
        return HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, top_k=config.top_k)
    else:
        raise ValueError(f"Unknown retriever: {config.retriever}")


# ── Orchestrator Assembly ─────────────────────────────────────────────────────

def _make_eval_orchestrator(
    retriever: Any,
    settings: Settings,
    hyde: bool = False,
) -> Any:
    """Build orchestrator wired to the eval retriever."""
    from src.agents.doc_agent import DocAgent
    from src.agents.orchestrator import Orchestrator
    from src.agents.simfin_agent import SimFinAgent
    from src.agents.synthesis_agent import SynthesisAgent
    from src.simfin.storage import read_table

    doc_agent = DocAgent(retriever=retriever)

    simfin_agent = SimFinAgent(settings=settings)

    # LLM client for synthesis
    llm_client = None
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        import anthropic
        llm_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    synthesis_agent = SynthesisAgent(llm_client=llm_client, model=settings.llm_model)

    # Load company map
    company_map = None
    map_path = Path(settings.simfin_processed_dir) / f"company_map.{settings.table_format}"
    if map_path.exists():
        company_map = read_table(map_path)

    return Orchestrator(
        doc_agent=doc_agent,
        simfin_agent=simfin_agent,
        synthesis_agent=synthesis_agent,
        company_map=company_map,
        hyde=hyde,
    )


# ── Multi-vectorstore build path ──────────────────────────────────────────────

def _get_or_build_index_for_subset(
    config: ExperimentConfig,
    chunks_df: pd.DataFrame,
    embedder: Any,
    subset_tag: str,
) -> Any:
    """Build a FAISS index for a named subset of chunks, with caching."""
    from src.vectorstore.faiss_store import FAISSStore

    idx_hash = _cache_hash(config) + f"_{subset_tag}"
    idx_dir = _INDEX_CACHE_DIR / idx_hash
    faiss_path = idx_dir / "faiss" / "index.faiss"

    if faiss_path.exists():
        store = FAISSStore(cache_dir=str(idx_dir), dim=embedder.dim)
        if store.count() > 0:
            logger.info("Using cached FAISS index [%s]: %s", subset_tag, idx_dir)
            return store
        logger.info("Cached index [%s] empty, rebuilding...", subset_tag)

    idx_dir.mkdir(parents=True, exist_ok=True)
    store = FAISSStore(cache_dir=str(idx_dir), dim=embedder.dim)
    texts = chunks_df["text"].fillna("").tolist()
    metadatas = chunks_df.to_dict(orient="records")
    logger.info("Embedding %d chunks [%s]...", len(texts), subset_tag)
    store.upsert(embedder.embed_bulk(texts), metadatas)
    logger.info("Built FAISS index [%s] with %d vectors at %s", subset_tag, store.count(), idx_dir)
    return store


def _build_multivector_pipeline(
    config: ExperimentConfig,
    settings: Settings,
) -> tuple[Any, pd.DataFrame]:
    """
    Build 3 specialized FAISS indexes for multi-vectorstore experiments.

    Stores created:
      transcript  — recursive chunks of transcripts
      sec_recursive — recursive chunks of SEC filings
      sec_section   — sec_section chunks of SEC filings
    """
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.multi_store import MultiStoreRetriever
    from src.simfin.storage import read_table

    docs_path = Path(settings.octus_processed_dir) / f"documents.{settings.table_format}"
    if not docs_path.exists():
        raise FileNotFoundError(
            f"Documents not found at {docs_path}. Run the bootstrap pipeline first."
        )
    docs_df = read_table(docs_path)

    # Build recursive chunks (transcript + SEC narrative)
    recursive_config = ExperimentConfig(
        name=config.name, chunker="recursive", retriever=config.retriever,
        embedding_model=config.embedding_model, top_k=config.top_k,
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
    )
    recursive_chunks = _get_or_build_chunks(recursive_config, docs_df, settings)
    transcript_chunks = recursive_chunks[
        recursive_chunks["doc_source"] == "transcript"
    ].reset_index(drop=True)
    sec_recursive_chunks = recursive_chunks[
        recursive_chunks["doc_source"] == "sec_filing"
    ].reset_index(drop=True)

    # Build sec_section chunks (SEC structural)
    sec_section_config = ExperimentConfig(
        name=config.name, chunker="sec_section", retriever=config.retriever,
        embedding_model=config.embedding_model, top_k=config.top_k,
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
    )
    all_sec_section = _get_or_build_chunks(sec_section_config, docs_df, settings)
    sec_section_chunks = all_sec_section[
        all_sec_section["doc_source"] == "sec_filing"
    ].reset_index(drop=True)

    embedder = _make_embedder(config)
    t_store = _get_or_build_index_for_subset(recursive_config, transcript_chunks, embedder, "transcript")
    sr_store = _get_or_build_index_for_subset(recursive_config, sec_recursive_chunks, embedder, "sec_recursive")
    ss_store = _get_or_build_index_for_subset(sec_section_config, sec_section_chunks, embedder, "sec_section")

    retriever = MultiStoreRetriever(
        transcript_retriever=DenseRetriever(vectorstore=t_store, embedder=embedder, top_k=config.top_k),
        sec_recursive_retriever=DenseRetriever(vectorstore=sr_store, embedder=embedder, top_k=config.top_k),
        sec_section_retriever=DenseRetriever(vectorstore=ss_store, embedder=embedder, top_k=config.top_k),
        top_k=config.top_k,
    )
    orchestrator = _make_eval_orchestrator(retriever, settings, hyde=config.hyde)
    return orchestrator, recursive_chunks


# ── Public API ────────────────────────────────────────────────────────────────

def build_eval_pipeline(
    config: ExperimentConfig,
    settings: Settings,
) -> tuple[Any, pd.DataFrame]:
    """
    Build isolated pipeline components for one experiment config.

    Returns:
        (orchestrator, chunks_df) — the orchestrator is ready to run queries,
        and chunks_df is needed for judge context.
    """
    if config.multivector:
        return _build_multivector_pipeline(config, settings)

    from src.simfin.storage import read_table

    docs_path = Path(settings.octus_processed_dir) / f"documents.{settings.table_format}"
    if not docs_path.exists():
        raise FileNotFoundError(
            f"Documents not found at {docs_path}. Run the bootstrap pipeline first "
            "(e.g., start the Streamlit app once to trigger ingestion)."
        )

    docs_df = read_table(docs_path)
    chunks_df = _get_or_build_chunks(config, docs_df, settings)

    embedder = _make_embedder(config)
    vectorstore = _get_or_build_index(config, chunks_df, embedder)
    retriever = _make_retriever(config, vectorstore, embedder, chunks_df)
    orchestrator = _make_eval_orchestrator(retriever, settings)

    return orchestrator, chunks_df


def run_single_experiment(
    config: ExperimentConfig,
    questions: list[EvalQuestion],
    settings: Settings,
) -> list[ExperimentResult]:
    """
    Run all questions through one experiment configuration.

    Returns a list of ExperimentResult, one per question.
    """
    logger.info("=== Experiment: %s ===", config.name)
    orchestrator, chunks_df = build_eval_pipeline(config, settings)

    results: list[ExperimentResult] = []
    for q in questions:
        logger.info("  Q [%s]: %s", q.id, q.text[:60])
        t_start = time.time()

        try:
            synthesis_result = orchestrator.run(q.text)
            t_total = (time.time() - t_start) * 1000

            # Serialize citations
            citations = []
            for c in synthesis_result.citations:
                citations.append({
                    k: v for k, v in vars(c).items()
                })

            result = ExperimentResult(
                experiment_name=config.name,
                question_id=q.id,
                config=config.to_dict(),
                citations=citations,
                answer_text=synthesis_result.final_answer_text,
                trace_events=synthesis_result.trace_events,
                timing={
                    "total_ms": round(t_total, 1),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.error("  Error on %s: %s", q.id, e)
            result = ExperimentResult(
                experiment_name=config.name,
                question_id=q.id,
                config=config.to_dict(),
                citations=[],
                answer_text=f"ERROR: {e}",
                trace_events=[],
                timing={"total_ms": (time.time() - t_start) * 1000},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        results.append(result)
        # Brief pause between synthesis calls to avoid per-minute API rate limits.
        time.sleep(1)
    return results


def _extract_chunks_from_trace(trace_events: list[dict]) -> list[dict]:
    """Extract retrieved chunk info from trace events."""
    for event in trace_events:
        if event.get("event_type") == "retrieval_results":
            payload = event.get("payload", {})
            chunks = payload.get("chunks", [])
            if chunks:
                return chunks
    return []
