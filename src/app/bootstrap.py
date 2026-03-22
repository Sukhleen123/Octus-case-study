"""
Bootstrap: ensure_ready(settings) -> AppContext

Takes the system from 'raw files only' -> 'processed artifacts + indexes + runtime objects ready'
in an idempotent, file-locked way. Safe to call repeatedly.

Definition:
  "Bootstrap" = the app's startup initializer. ensure_ready(settings) checks if
  processed artifacts match the current inputs (via fingerprint). If they do, it
  loads existing artifacts without rebuilding. If not, it runs the full pipeline:
  ingestion → chunking → embedding → indexing → runtime object construction.

Usage:
    # In Streamlit (cached across reruns):
    @st.cache_resource
    def get_app_context():
        return ensure_ready(Settings())

    # In Chainlit / FastAPI:
    ctx = ensure_ready(Settings())
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.context import AppContext
from src.app.fingerprints import compute_fingerprint
from src.app.locks import acquire_lock
from src.app.logging import configure_logging
from src.app.settings import Settings

logger = logging.getLogger(__name__)

_MANIFEST_FILE = "bootstrap_manifest.json"
_LOCK_FILE = "data/processed/.bootstrap.lock"


# ── Manifest helpers ───────────────────────────────────────────────────────────

def _read_manifest(manifest_path: Path) -> dict | None:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_manifest(manifest_path: Path, fingerprint: str, settings: Settings) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fingerprint": fingerprint,
        "built_at": datetime.utcnow().isoformat(),
        "settings_subset": settings.fingerprint_subset(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _artifacts_exist(settings: Settings) -> bool:
    """Return True if all expected output files exist."""
    out = settings.octus_processed_path
    fmt = settings.table_format
    required = [
        out / f"documents.{fmt}",
        out / f"chunks_heading.{fmt}",
        out / f"chunks_token.{fmt}",
    ]
    return all(p.exists() for p in required)


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def _run_pipeline(settings: Settings) -> None:
    """Run full ingestion + chunking + embedding + indexing."""
    from src.chunking.heading_chunker import HeadingChunker
    from src.chunking.token_chunker import TokenChunker
    from src.embeddings.embedder import get_embedder
    from src.octus.ingest import run_ingestion
    from src.simfin.mapping import build_company_map
    from src.simfin.metrics_catalog import build_metrics_catalog

    t0 = time.time()
    logger.info("Bootstrap: starting full pipeline build")

    # Step 1: Octus ingestion
    docs_df = run_ingestion(settings)
    logger.info("Ingestion complete: %d documents", len(docs_df))

    # Step 2: Chunking (always build both)
    _build_chunks(settings, docs_df)

    # Step 3: SimFin mapping + batch ingest
    company_map = build_company_map(settings)
    build_metrics_catalog(settings)

    if settings.simfin_mode == "batch" and settings.simfin_api_key:
        from src.simfin.batch_ingest import run_batch_ingest
        run_batch_ingest(settings, company_map)

    # Step 4: Embed + index chunks
    embedder = get_embedder(settings)
    vectorstore = _build_vectorstore(settings, embedder)

    logger.info("Bootstrap pipeline complete in %.1fs", time.time() - t0)


def _build_chunks(settings: Settings, docs_df: pd.DataFrame) -> None:
    """Run both chunkers and write output tables."""
    from src.chunking.heading_chunker import HeadingChunker
    from src.chunking.token_chunker import TokenChunker
    from src.simfin.storage import write_table

    out_dir = settings.octus_processed_path
    fmt = settings.table_format

    chunkers = {
        "heading": HeadingChunker(),
        "token": TokenChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
    }

    for chunker_id, chunker in chunkers.items():
        if chunker_id not in settings.chunkers_to_build:
            continue
        logger.info("Chunking with %s chunker (%d docs)...", chunker_id, len(docs_df))
        all_chunks = []
        for _, row in docs_df.iterrows():
            doc = row.to_dict()
            chunks = chunker.chunk(doc)
            all_chunks.extend([c.to_dict() for c in chunks])

        chunks_df = pd.DataFrame(all_chunks)
        out_path = out_dir / f"chunks_{chunker_id}.{fmt}"
        write_table(chunks_df, out_path, fmt)
        logger.info("Wrote %d chunks to %s", len(chunks_df), out_path)


def _build_vectorstore(settings: Settings, embedder: Any) -> Any:
    """Build and populate the vectorstore from the selected chunker's output."""
    from src.simfin.storage import read_table

    out_dir = settings.octus_processed_path
    fmt = settings.table_format

    # Use the first available chunker as the active one for indexing
    active_chunker = settings.chunkers_to_build[0] if settings.chunkers_to_build else "token"
    chunks_path = out_dir / f"chunks_{active_chunker}.{fmt}"

    if not chunks_path.exists():
        logger.warning("Chunks file not found: %s — skipping indexing", chunks_path)
        return _make_vectorstore(settings, embedder)

    chunks_df = read_table(chunks_path)
    logger.info("Embedding %d chunks with %s embedder...", len(chunks_df), settings.embedding_provider)

    vectorstore = _make_vectorstore(settings, embedder)

    texts = chunks_df["text"].fillna("").tolist()
    metadatas = chunks_df.to_dict(orient="records")

    # Embed in batches of 100
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        vectors = embedder.embed(batch_texts)
        vectorstore.upsert(vectors, batch_meta)
        logger.debug("Indexed batch %d-%d", i, i + batch_size)

    logger.info("Vectorstore populated with %d vectors", vectorstore.count())
    return vectorstore


def _make_vectorstore(settings: Settings, embedder: Any) -> Any:
    """Create (but don't populate) the appropriate vectorstore."""
    if settings.use_pinecone():
        from src.vectorstore.metadata_store import MetadataStore
        from src.vectorstore.pinecone_store import PineconeStore
        from pathlib import Path

        # Local SQLite store holds chunk text (excluded from Pinecone due to 40 KB limit)
        text_store = MetadataStore(
            Path(settings.cache_dir) / "pinecone" / "text_store.sqlite"
        )
        return PineconeStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace,
            dim=settings.embedding_dim,
            text_store=text_store,
        )
    else:
        from src.vectorstore.faiss_store import FAISSStore
        return FAISSStore(
            cache_dir=settings.cache_dir,
            dim=settings.embedding_dim,
        )


def _make_retriever(settings: Settings, vectorstore: Any, embedder: Any) -> Any:
    """Instantiate the configured retriever."""
    if settings.retriever_id == "dense_mmr":
        from src.retrieval.dense_mmr import DenseMMRRetriever
        return DenseMMRRetriever(
            vectorstore=vectorstore,
            embedder=embedder,
            top_k=settings.top_k,
            mmr_lambda=settings.mmr_lambda,
        )
    else:
        from src.retrieval.dense import DenseRetriever
        return DenseRetriever(
            vectorstore=vectorstore,
            embedder=embedder,
            top_k=settings.top_k,
        )


def _make_simfin_client(settings: Settings) -> Any | None:
    """Create SimFin client (v3 realtime or None for batch-only mode)."""
    if settings.simfin_mode != "realtime" or not settings.simfin_api_key:
        return None
    from src.simfin.cache import RealtimeCache
    from src.simfin.realtime_client_v3 import SimFinV3Client
    cache = RealtimeCache(cache_dir=settings.cache_dir)
    return SimFinV3Client(
        api_key=settings.simfin_api_key,
        cache=cache,
        base_url=settings.simfin_base_url,
    )


def _make_llm_client(settings: Settings) -> Any | None:
    """Create Anthropic client if configured, else None (mock mode)."""
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return None


def _make_orchestrator(
    settings: Settings,
    retriever: Any,
    simfin_client: Any,
    company_map: pd.DataFrame | None,
) -> Any:
    """Assemble the orchestrator from its component agents."""
    from src.agents.doc_agent import DocAgent
    from src.agents.orchestrator import Orchestrator
    from src.agents.simfin_agent import SimFinAgent
    from src.agents.synthesis_agent import SynthesisAgent

    doc_agent = DocAgent(retriever=retriever)
    simfin_agent = SimFinAgent(simfin_client=simfin_client, settings=settings)
    llm_client = _make_llm_client(settings)
    synthesis_agent = SynthesisAgent(llm_client=llm_client)

    return Orchestrator(
        doc_agent=doc_agent,
        simfin_agent=simfin_agent,
        synthesis_agent=synthesis_agent,
        company_map=company_map,
    )


# ── Main entry point ───────────────────────────────────────────────────────────

def ensure_ready(settings: Settings) -> AppContext:
    """
    Idempotent bootstrap. Safe to call on every Streamlit rerun.

    1. Compute fingerprint of raw files + settings
    2. Acquire file lock (prevents parallel builds)
    3. Compare fingerprint to stored manifest
    4. If mismatch or missing artifacts: run full pipeline
    5. Load / create runtime objects (vectorstore, retriever, etc.)
    6. Return AppContext

    The auto-trigger behavior (settings.apply_mode == "auto") is the default.
    Any change to tracked settings fields will trigger a rebuild automatically.
    """
    configure_logging()

    manifest_path = settings.octus_processed_path / _MANIFEST_FILE
    fingerprint = compute_fingerprint(settings)

    with acquire_lock(_LOCK_FILE, timeout=120.0):
        manifest = _read_manifest(manifest_path)
        already_built = (
            manifest is not None
            and manifest.get("fingerprint") == fingerprint
            and _artifacts_exist(settings)
        )

        if already_built:
            logger.info("Bootstrap: artifacts up-to-date (fingerprint match). Skipping rebuild.")
        else:
            reason = "fingerprint mismatch" if manifest else "first run"
            logger.info("Bootstrap: rebuilding artifacts (%s).", reason)
            _run_pipeline(settings)
            _write_manifest(manifest_path, fingerprint, settings)

    # ── Construct runtime objects ───────────────────────────────────────────
    from src.embeddings.embedder import get_embedder

    embedder = get_embedder(settings)
    vectorstore = _make_vectorstore(settings, embedder)
    retriever = _make_retriever(settings, vectorstore, embedder)
    simfin_client = _make_simfin_client(settings)

    # Load company_map if it exists
    company_map = None
    map_path = settings.simfin_processed_path / f"company_map.{settings.table_format}"
    if map_path.exists():
        from src.simfin.storage import read_table
        company_map = read_table(map_path)

    orchestrator = _make_orchestrator(settings, retriever, simfin_client, company_map)

    logger.info("Bootstrap complete. Vectorstore type: %s", type(vectorstore).__name__)

    return AppContext(
        vectorstore=vectorstore,
        retriever=retriever,
        simfin_client=simfin_client,
        orchestrator=orchestrator,
        settings_fingerprint=fingerprint,
    )
