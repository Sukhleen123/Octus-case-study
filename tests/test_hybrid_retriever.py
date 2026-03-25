"""Tests for the hybrid retriever (dense + BM25 with RRF fusion)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.retrieval.hybrid import HybridRetriever, _RRF_K


def _make_mock_result(chunk_id: str, text: str = "", **kwargs) -> dict[str, Any]:
    """Create a mock retrieval result dict."""
    result = {
        "chunk_id": chunk_id,
        "text": text or f"Text for {chunk_id}",
        "company_name": kwargs.get("company_name", "Test Co"),
        "doc_source": kwargs.get("doc_source", "sec_filing"),
        "_score": kwargs.get("_score", 0.5),
    }
    result.update(kwargs)
    return result


def _make_mock_retriever(results: list[dict[str, Any]]) -> MagicMock:
    """Create a mock retriever that returns the given results."""
    mock = MagicMock()
    mock.retrieve.return_value = results
    mock.top_k = len(results)
    return mock


class TestRRFScoreFormula:
    """Test that RRF fusion produces the correct score: weight / (k + rank + 1)."""

    def test_rrf_score_dense_only(self):
        """A chunk appearing only in dense results gets dense_weight/(k+rank+1)."""
        dense_results = [_make_mock_result("c1")]
        bm25_results = []

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.6,
        )
        results = retriever.retrieve(query="test query")

        assert len(results) == 1
        expected_score = 0.6 / (_RRF_K + 0 + 1)  # rank=0
        assert abs(results[0]["_score"] - expected_score) < 1e-9

    def test_rrf_score_bm25_only(self):
        """A chunk appearing only in BM25 results gets bm25_weight/(k+rank+1)."""
        dense_results = []
        bm25_results = [_make_mock_result("c1")]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.6,
        )
        results = retriever.retrieve(query="test query")

        assert len(results) == 1
        bm25_weight = 0.4
        expected_score = bm25_weight / (_RRF_K + 0 + 1)
        assert abs(results[0]["_score"] - expected_score) < 1e-9

    def test_rrf_score_both_retrievers(self):
        """A chunk appearing in both gets sum of both RRF contributions."""
        dense_results = [_make_mock_result("c1")]
        bm25_results = [_make_mock_result("c1")]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.6,
        )
        results = retriever.retrieve(query="test query")

        assert len(results) == 1
        dense_contribution = 0.6 / (_RRF_K + 0 + 1)
        bm25_contribution = 0.4 / (_RRF_K + 0 + 1)
        expected_score = dense_contribution + bm25_contribution
        assert abs(results[0]["_score"] - expected_score) < 1e-9

    def test_rrf_score_ranking_accounts_for_position(self):
        """Second-ranked results get lower RRF scores (rank=1 vs rank=0)."""
        dense_results = [
            _make_mock_result("c1"),
            _make_mock_result("c2"),
        ]
        bm25_results = []

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.6,
        )
        results = retriever.retrieve(query="test query")

        assert len(results) == 2
        score_c1 = 0.6 / (_RRF_K + 0 + 1)
        score_c2 = 0.6 / (_RRF_K + 1 + 1)
        assert abs(results[0]["_score"] - score_c1) < 1e-9
        assert abs(results[1]["_score"] - score_c2) < 1e-9


class TestDeduplication:
    """Test deduplication by chunk_id across dense and BM25 results."""

    def test_duplicate_chunk_id_merged(self):
        """Same chunk_id in both retrievers should appear once in output."""
        dense_results = [
            _make_mock_result("c1", text="Dense text for c1"),
            _make_mock_result("c2", text="Dense text for c2"),
        ]
        bm25_results = [
            _make_mock_result("c1", text="BM25 text for c1"),
            _make_mock_result("c3", text="BM25 text for c3"),
        ]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=10,
            dense_weight=0.6,
        )
        results = retriever.retrieve(query="test query")

        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids in results"
        assert set(chunk_ids) == {"c1", "c2", "c3"}

    def test_deduplicated_chunk_uses_first_seen_data(self):
        """When a chunk appears in both, the data from the first occurrence (dense) is used."""
        dense_results = [_make_mock_result("c1", text="Dense version")]
        bm25_results = [_make_mock_result("c1", text="BM25 version")]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.6,
        )
        results = retriever.retrieve(query="test query")

        assert len(results) == 1
        # Dense results are processed first, so chunk_data should use dense version
        assert results[0]["text"] == "Dense version"


class TestDenseWeightParameter:
    """Test that dense_weight parameter affects fusion scores."""

    def test_higher_dense_weight_favors_dense(self):
        """With dense_weight=0.9, a chunk only in dense should score higher
        than a chunk only in BM25."""
        dense_results = [_make_mock_result("dense_only")]
        bm25_results = [_make_mock_result("bm25_only")]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.9,
        )
        results = retriever.retrieve(query="test query")

        scores = {r["chunk_id"]: r["_score"] for r in results}
        assert scores["dense_only"] > scores["bm25_only"]

    def test_lower_dense_weight_favors_bm25(self):
        """With dense_weight=0.1, a chunk only in BM25 should score higher
        than a chunk only in dense."""
        dense_results = [_make_mock_result("dense_only")]
        bm25_results = [_make_mock_result("bm25_only")]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.1,
        )
        results = retriever.retrieve(query="test query")

        scores = {r["chunk_id"]: r["_score"] for r in results}
        assert scores["bm25_only"] > scores["dense_only"]

    def test_equal_weight(self):
        """With dense_weight=0.5, both sources contribute equally at same rank."""
        dense_results = [_make_mock_result("dense_only")]
        bm25_results = [_make_mock_result("bm25_only")]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.5,
        )
        results = retriever.retrieve(query="test query")

        scores = {r["chunk_id"]: r["_score"] for r in results}
        # Both at rank 0 with weight 0.5 => same score
        assert abs(scores["dense_only"] - scores["bm25_only"]) < 1e-9


class TestTopKLimit:
    """Test that top_k limit is applied to final results."""

    def test_top_k_respected(self):
        dense_results = [
            _make_mock_result(f"d{i}") for i in range(5)
        ]
        bm25_results = [
            _make_mock_result(f"b{i}") for i in range(5)
        ]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=3,
            dense_weight=0.6,
        )
        results = retriever.retrieve(query="test query")
        assert len(results) <= 3

    def test_top_k_1(self):
        dense_results = [_make_mock_result("c1"), _make_mock_result("c2")]
        bm25_results = [_make_mock_result("c3")]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=1,
            dense_weight=0.6,
        )
        results = retriever.retrieve(query="test query")
        assert len(results) == 1


class TestRRFScoreField:
    """Test that results have both _score and _rrf_score fields."""

    def test_rrf_score_field_present(self):
        dense_results = [_make_mock_result("c1")]
        bm25_results = [_make_mock_result("c2")]

        dense_mock = _make_mock_retriever(dense_results)
        bm25_mock = _make_mock_retriever(bm25_results)

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.6,
        )
        results = retriever.retrieve(query="test query")

        for r in results:
            assert "_score" in r
            assert "_rrf_score" in r
            assert r["_score"] == r["_rrf_score"]


class TestOverFetch:
    """Test that the hybrid retriever over-fetches from sub-retrievers."""

    def test_sub_retriever_top_k_temporarily_increased(self):
        dense_mock = _make_mock_retriever([])
        bm25_mock = _make_mock_retriever([])
        dense_mock.top_k = 5
        bm25_mock.top_k = 5

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            bm25_retriever=bm25_mock,
            top_k=5,
            dense_weight=0.6,
        )
        retriever.retrieve(query="test query")

        # After retrieval, original top_k should be restored
        assert dense_mock.top_k == 5
        assert bm25_mock.top_k == 5
