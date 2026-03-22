"""
SentenceTransformer embedder — uses intfloat/e5-base-v2 by default.

E5 models require task-specific prefixes:
  - Documents (indexing): "passage: <text>"
  - Queries (retrieval):  "query: <text>"

Without these prefixes retrieval quality degrades significantly.
See: https://huggingface.co/intfloat/e5-base-v2

Usage:
    embedder = SentenceTransformerEmbedder("intfloat/e5-base-v2")
    # At index time:
    doc_vecs = embedder.embed(texts)           # adds "passage: " prefix
    # At query time:
    q_vec = embedder.embed_query("revenue?")   # adds "query: " prefix
"""

from __future__ import annotations

from src.embeddings.embedder import BaseEmbedder

_DEFAULT_MODEL = "intfloat/e5-base-v2"
_E5_MODELS = {"e5-base-v2", "e5-small-v2", "e5-large-v2", "e5-base", "e5-small", "e5-large"}


def _needs_prefix(model_name: str) -> bool:
    """Return True if the model name is an E5 variant that requires prefixes."""
    return any(e5 in model_name for e5 in _E5_MODELS)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder backed by a local SentenceTransformer model.

    For E5 models (intfloat/e5-*) the correct prefixes are added
    automatically to both passage and query embeddings.

    Args:
        model_name: HuggingFace model identifier (default: intfloat/e5-base-v2).
        batch_size: Batch size for encode() calls.
        normalize: Whether to L2-normalize output vectors (True for cosine similarity).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            ) from e

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize
        self._use_prefix = _needs_prefix(model_name)

        # Determine dim from model config
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of passage texts (for indexing).

        E5 models: prepends "passage: " to each text.
        """
        if self._use_prefix:
            texts = [f"passage: {t}" for t in texts]

        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query string for retrieval.

        E5 models: prepends "query: " — distinct from passage prefix.
        This method should be used by retrievers instead of embed_one().
        """
        if self._use_prefix:
            query = f"query: {query}"

        vec = self._model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return vec[0].tolist()

    def embed_one(self, text: str) -> list[float]:
        """
        Embed a single text as a passage (for indexing).
        Use embed_query() at retrieval time.
        """
        return self.embed([text])[0]
