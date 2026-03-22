"""
BaseEmbedder ABC and get_embedder() factory.

Usage:
    from src.embeddings.embedder import get_embedder
    embedder = get_embedder(settings)
    vectors = embedder.embed(["text one", "text two"])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app.settings import Settings


class BaseEmbedder(ABC):
    """Abstract interface for all embedding providers."""

    dim: int  # embedding dimension — must be set by subclasses

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts.

        Args:
            texts: Non-empty list of strings.

        Returns:
            List of embedding vectors (same length as texts).
        """
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single text."""
        return self.embed([text])[0]


def get_embedder(settings: "Settings") -> BaseEmbedder:
    """
    Factory: return the appropriate embedder based on settings.embedding_provider.

    Providers:
      "mock"                 — deterministic random vectors, no API key needed
      "sentence_transformers"— local HuggingFace model (e.g. intfloat/e5-base-v2)
      "openai"               — OpenAI embeddings API
    """
    provider = settings.embedding_provider

    if provider == "mock":
        from src.embeddings.mock_embedder import MockEmbedder
        return MockEmbedder(dim=settings.embedding_dim)

    if provider == "sentence_transformers":
        from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
        model_name = settings.embedding_model or "intfloat/e5-base-v2"
        return SentenceTransformerEmbedder(model_name=model_name)

    if provider == "openai":
        try:
            from src.embeddings.openai_embedder import OpenAIEmbedder
            return OpenAIEmbedder(
                model=settings.embedding_model or "text-embedding-3-small",
                dim=settings.embedding_dim,
            )
        except ImportError as e:
            raise ImportError(
                "openai package is required for embedding_provider='openai'. "
                "Run: pip install openai"
            ) from e

    raise ValueError(
        f"Unsupported embedding_provider: {provider!r}. "
        "Allowed: 'mock', 'sentence_transformers', 'openai'."
    )
