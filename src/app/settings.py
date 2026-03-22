"""
Centralized application settings.

Load order:
  1. configs/app.yaml  (defaults)
  2. Environment variables (overrides, e.g. SIMFIN_API_KEY)

Usage:
    from src.app.settings import Settings
    settings = Settings()
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Path to the default config file, relative to the project root.
_CONFIG_PATH = Path("configs/app.yaml")


def _load_yaml_defaults() -> dict:
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open() as f:
            return yaml.safe_load(f) or {}
    return {}


_YAML_DEFAULTS = _load_yaml_defaults()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    octus_raw_dir: str = Field(
        default=_YAML_DEFAULTS.get("octus_raw_dir", "data/raw/octus")
    )
    processed_dir: str = Field(
        default=_YAML_DEFAULTS.get("processed_dir", "data/processed")
    )
    cache_dir: str = Field(
        default=_YAML_DEFAULTS.get("cache_dir", "data/cache")
    )

    # ── Storage ────────────────────────────────────────────────────────────────
    table_format: Literal["parquet", "csv"] = Field(
        default=_YAML_DEFAULTS.get("table_format", "parquet")
    )
    duckdb_path: str = Field(
        default=_YAML_DEFAULTS.get("duckdb_path", "data/processed/simfin/simfin.duckdb")
    )

    # ── SimFin ─────────────────────────────────────────────────────────────────
    simfin_api_key: str = Field(default=_YAML_DEFAULTS.get("simfin_api_key", ""))
    simfin_base_url: str = Field(
        default=_YAML_DEFAULTS.get("simfin_base_url", "https://backend.simfin.com")
    )
    simfin_mode: Literal["batch", "realtime"] = Field(
        default=_YAML_DEFAULTS.get("simfin_mode", "batch")
    )
    simfin_periodicity: Literal["annual", "quarterly", "both"] = Field(
        default=_YAML_DEFAULTS.get("simfin_periodicity", "annual")
    )

    # ── Mapping ────────────────────────────────────────────────────────────────
    mapping_mode: Literal["auto_matched", "confirmed", "both"] = Field(
        default=_YAML_DEFAULTS.get("mapping_mode", "auto_matched")
    )
    auto_promote_matched: bool = Field(
        default=_YAML_DEFAULTS.get("auto_promote_matched", True)
    )

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunkers_to_build: list[str] = Field(
        default=_YAML_DEFAULTS.get("chunkers_to_build", ["heading", "token"])
    )
    chunk_size: int = Field(default=_YAML_DEFAULTS.get("chunk_size", 512))
    chunk_overlap: int = Field(default=_YAML_DEFAULTS.get("chunk_overlap", 64))

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retriever_id: Literal["dense", "dense_mmr"] = Field(
        default=_YAML_DEFAULTS.get("retriever_id", "dense")
    )
    mmr_lambda: float = Field(default=_YAML_DEFAULTS.get("mmr_lambda", 0.5))
    top_k: int = Field(default=_YAML_DEFAULTS.get("top_k", 5))

    # ── Vectorstore ────────────────────────────────────────────────────────────
    vectorstore_provider: Literal["auto", "pinecone", "faiss"] = Field(
        default=_YAML_DEFAULTS.get("vectorstore_provider", "auto")
    )
    pinecone_api_key: str = Field(default=_YAML_DEFAULTS.get("pinecone_api_key", ""))
    pinecone_index_name: str = Field(
        default=_YAML_DEFAULTS.get("pinecone_index_name", "octus-financial")
    )
    pinecone_namespace: str = Field(
        default=_YAML_DEFAULTS.get("pinecone_namespace", "default")
    )

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_provider: Literal["mock", "sentence_transformers", "openai", "anthropic"] = Field(
        default=_YAML_DEFAULTS.get("embedding_provider", "mock")
    )
    embedding_model: str = Field(default=_YAML_DEFAULTS.get("embedding_model", ""))
    embedding_dim: int = Field(default=_YAML_DEFAULTS.get("embedding_dim", 1536))

    # ── LLM (synthesis agent) ──────────────────────────────────────────────────
    llm_provider: Literal["none", "anthropic"] = Field(
        default=_YAML_DEFAULTS.get("llm_provider", "none")
    )
    llm_model: str = Field(
        default=_YAML_DEFAULTS.get("llm_model", "claude-sonnet-4-6")
    )
    anthropic_api_key: str = Field(default=_YAML_DEFAULTS.get("anthropic_api_key", ""))

    # ── UI ─────────────────────────────────────────────────────────────────────
    apply_mode: Literal["auto", "manual"] = Field(
        default=_YAML_DEFAULTS.get("apply_mode", "auto")
    )

    # ── SEC HTML verification ──────────────────────────────────────────────────
    require_sec_html: bool = Field(default=_YAML_DEFAULTS.get("require_sec_html", True))

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def octus_raw_path(self) -> Path:
        return Path(self.octus_raw_dir)

    @property
    def processed_path(self) -> Path:
        return Path(self.processed_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def octus_processed_path(self) -> Path:
        return self.processed_path / "octus"

    @property
    def simfin_processed_path(self) -> Path:
        return self.processed_path / "simfin"

    @field_validator("chunkers_to_build")
    @classmethod
    def _validate_chunkers(cls, v: list[str]) -> list[str]:
        allowed = {"heading", "token"}
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"Invalid chunkers: {bad}. Allowed: {allowed}")
        return v

    def use_pinecone(self) -> bool:
        """Return True if Pinecone should be used as the vectorstore."""
        if self.vectorstore_provider == "faiss":
            return False
        if self.vectorstore_provider == "pinecone":
            return True
        # "auto": use Pinecone only if key is set
        return bool(self.pinecone_api_key)

    def fingerprint_subset(self) -> dict:
        """Return the settings fields that affect processed artifacts."""
        return {
            "table_format": self.table_format,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunkers_to_build": sorted(self.chunkers_to_build),
            "mapping_mode": self.mapping_mode,
            "auto_promote_matched": self.auto_promote_matched,
            "simfin_mode": self.simfin_mode,
            "simfin_periodicity": self.simfin_periodicity,
            "vectorstore_provider": self.vectorstore_provider,
        }
