"""
Centralized application settings.

Load order:
  1. Environment variables from .env (single source of truth)
  2. Field defaults below (fallback)

Usage:
    from src.app.settings import Settings
    settings = Settings()
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    octus_raw_dir: str = Field(default="data/raw/octus")
    octus_processed_dir: str = Field(default="data/processed/octus")
    simfin_processed_dir: str = Field(default="data/processed/simfin")
    cache_dir: str = Field(default="data/cache")

    # ── Storage ────────────────────────────────────────────────────────────────
    table_format: Literal["parquet", "csv"] = Field(default="parquet")
    duckdb_path: str = Field(default="data/processed/simfin/simfin.duckdb")

    # ── SimFin ─────────────────────────────────────────────────────────────────
    simfin_api_key: str = Field(default="")
    simfin_base_url: str = Field(default="https://backend.simfin.com")

    # ── Mapping ────────────────────────────────────────────────────────────────
    mapping_mode: Literal["auto_matched", "confirmed", "both"] = Field(default="both")
    auto_promote_matched: bool = Field(default=True)

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retriever_id: Literal["dense", "dense_mmr"] = Field(default="dense")
    mmr_lambda: float = Field(default=0.5)
    top_k: int = Field(default=10)

    # ── Vectorstore / Pinecone ─────────────────────────────────────────────────
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="octus-financial")
    pinecone_transcripts_namespace: str = Field(default="transcripts")
    pinecone_sec_namespace: str = Field(default="sec_filings")

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_provider: Literal["sentence_transformers", "openai", "anthropic"] = Field(
        default="sentence_transformers"
    )
    embedding_model: str = Field(default="")
    embedding_dim: int = Field(default=768)

    # ── LLM (synthesis agent) ──────────────────────────────────────────────────
    llm_provider: Literal["none", "anthropic"] = Field(default="none")
    llm_model: str = Field(default="claude-sonnet-4-6")
    anthropic_api_key: str = Field(default="")

    # ── Agent ──────────────────────────────────────────────────────────────────
    hyde: bool = True
    doc_agent_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Model used by the doc agent for tool selection and chunk selection.",
    )

    # ── SEC HTML verification ──────────────────────────────────────────────────
    require_sec_html: bool = Field(default=True)
