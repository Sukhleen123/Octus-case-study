"""
Octus document ingestion pipeline.

Ingests raw Octus files, chunks them with document-type-appropriate chunkers,
embeds the chunks, and upserts into two Pinecone namespaces:

  transcripts → RecursiveChunker  → settings.pinecone_transcripts_namespace
  sec_filings → SECSectionChunker → settings.pinecone_sec_namespace

This is a full replace: each namespace is cleared before re-indexing.

Usage:
    from src.ingest.octus import ingest_octus
    ingest_octus(settings)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.settings import Settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


def ingest_octus(settings: Settings) -> None:
    """
    Run the full Octus ingestion pipeline:
      1. Parse raw JSON files → documents DataFrame
      2. Split by doc_source (transcript vs. sec_filing)
      3. Chunk each group with the appropriate chunker
      4. Embed + upsert into the corresponding Pinecone namespace (full replace)
    """
    from src.chunking.recursive_chunker import RecursiveChunker
    from src.chunking.sec_section_chunker import SECSectionChunker
    from src.embeddings.embedder import get_embedder
    from src.octus.ingest import run_ingestion
    from src.vectorstore.metadata_store import MetadataStore
    from src.vectorstore.pinecone_store import PineconeStore

    logger.info("Starting Octus ingestion...")
    docs_df = run_ingestion(settings)
    logger.info("Ingested %d documents", len(docs_df))

    embedder = get_embedder(settings)
    cache_dir = Path(settings.cache_dir) / "pinecone"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Transcripts → RecursiveChunker ────────────────────────────────────────
    transcript_rows = docs_df[docs_df["doc_source"] == "transcript"]
    if not transcript_rows.empty:
        logger.info("Chunking %d transcripts with RecursiveChunker...", len(transcript_rows))
        chunker = RecursiveChunker(
            target_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        transcript_chunks = _chunk_docs(chunker, transcript_rows)
        logger.info("Produced %d transcript chunks", len(transcript_chunks))

        transcript_store = PineconeStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            namespace=settings.pinecone_transcripts_namespace,
            dim=settings.embedding_dim,
            text_store=MetadataStore(cache_dir / "transcripts_text_store.sqlite"),
        )
        logger.info("Clearing transcripts namespace for full refresh...")
        transcript_store.clear()
        _embed_and_upsert(embedder, transcript_store, transcript_chunks)
        logger.info(
            "Transcripts namespace populated: %d vectors", transcript_store.count()
        )
    else:
        logger.warning("No transcript documents found — skipping transcripts namespace")

    # ── SEC Filings → SECSectionChunker ───────────────────────────────────────
    sec_rows = docs_df[docs_df["doc_source"] == "sec_filing"]
    if not sec_rows.empty:
        logger.info("Chunking %d SEC filings with SECSectionChunker...", len(sec_rows))
        sec_chunker = SECSectionChunker()
        sec_chunks = _chunk_docs(sec_chunker, sec_rows)
        logger.info("Produced %d SEC filing chunks", len(sec_chunks))

        sec_store = PineconeStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            namespace=settings.pinecone_sec_namespace,
            dim=settings.embedding_dim,
            text_store=MetadataStore(cache_dir / "sec_filings_text_store.sqlite"),
        )
        logger.info("Clearing sec_filings namespace for full refresh...")
        sec_store.clear()
        _embed_and_upsert(embedder, sec_store, sec_chunks)
        logger.info(
            "SEC filings namespace populated: %d vectors", sec_store.count()
        )
    else:
        logger.warning("No SEC filing documents found — skipping sec_filings namespace")


def _chunk_docs(chunker: Any, docs_df: pd.DataFrame) -> list[dict]:
    """Run a chunker over every row in docs_df and return all chunk dicts."""
    all_chunks = []
    for _, row in docs_df.iterrows():
        doc = row.to_dict()
        try:
            chunks = chunker.chunk(doc)
            all_chunks.extend([c.to_dict() for c in chunks])
        except Exception as e:
            logger.warning(
                "Chunking failed for doc %s: %s", doc.get("document_id", "?"), e
            )
    return all_chunks


def _embed_and_upsert(embedder: Any, store: Any, chunks: list[dict]) -> None:
    """Embed chunks in batches and upsert into the vectorstore."""
    texts = [c.get("text", "") for c in chunks]
    for i in range(0, len(texts), _BATCH_SIZE):
        batch_texts = texts[i : i + _BATCH_SIZE]
        batch_meta = chunks[i : i + _BATCH_SIZE]
        vectors = embedder.embed(batch_texts)
        store.upsert(vectors, batch_meta)
        logger.debug("Upserted batch %d–%d", i, i + _BATCH_SIZE)
