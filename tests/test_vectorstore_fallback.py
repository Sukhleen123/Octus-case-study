"""Test that FAISS is used when Pinecone API key is not set."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch


class TestVectorstoreFallback:
    def test_faiss_when_no_pinecone_key(self):
        """With no pinecone_api_key and vectorstore_provider='auto', FAISSStore is used."""
        from src.app.settings import Settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                pinecone_api_key="",
                vectorstore_provider="auto",
                embedding_provider="mock",
                cache_dir=tmpdir,
            )

            from src.embeddings.mock_embedder import MockEmbedder
            embedder = MockEmbedder(dim=settings.embedding_dim)

            from src.app.bootstrap import _make_vectorstore
            from src.vectorstore.faiss_store import FAISSStore

            store = _make_vectorstore(settings, embedder)
            assert isinstance(store, FAISSStore)

    def test_pinecone_when_key_set(self):
        """With pinecone_api_key set, PineconeStore is attempted."""
        from src.app.settings import Settings

        settings = Settings(
            pinecone_api_key="test-key-12345",
            vectorstore_provider="auto",
            embedding_provider="mock",
        )

        assert settings.use_pinecone() is True

    def test_faiss_forced_when_provider_is_faiss(self):
        """vectorstore_provider='faiss' always uses FAISS even if key is set."""
        from src.app.settings import Settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                pinecone_api_key="some-key",
                vectorstore_provider="faiss",
                embedding_provider="mock",
                cache_dir=tmpdir,
            )

            assert settings.use_pinecone() is False

            from src.embeddings.mock_embedder import MockEmbedder
            from src.app.bootstrap import _make_vectorstore
            from src.vectorstore.faiss_store import FAISSStore

            embedder = MockEmbedder(dim=settings.embedding_dim)
            store = _make_vectorstore(settings, embedder)
            assert isinstance(store, FAISSStore)

    def test_faiss_upsert_and_query(self):
        """FAISSStore can upsert and query vectors without external dependencies."""
        from src.vectorstore.faiss_store import FAISSStore
        from src.embeddings.mock_embedder import MockEmbedder

        with tempfile.TemporaryDirectory() as tmpdir:
            dim = 64
            embedder = MockEmbedder(dim=dim)
            store = FAISSStore(cache_dir=tmpdir, dim=dim)

            texts = ["Hello world", "Financial results Q4", "Revenue grew 15%"]
            vectors = embedder.embed(texts)
            metadatas = [
                {"chunk_id": f"chunk_{i}", "text": t, "doc_source": "transcript"}
                for i, t in enumerate(texts)
            ]
            store.upsert(vectors, metadatas)
            assert store.count() == 3

            query_vec = embedder.embed_one("financial revenue")
            results = store.query_with_filter(query_vec, k=2)
            assert len(results) <= 2
            assert all("chunk_id" in r for r in results)
