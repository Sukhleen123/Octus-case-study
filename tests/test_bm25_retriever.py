"""Tests for the BM25 retriever."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.retrieval.bm25 import BM25Retriever

# Simple test chunks with known text for predictable BM25 scoring.
# Each chunk has distinct keywords to test retrieval specificity.
TEST_CHUNKS = [
    {
        "chunk_id": "chunk_001",
        "document_id": "doc_001",
        "doc_source": "sec_filing",
        "document_type": "10-K",
        "document_date": datetime(2024, 3, 15),
        "company_name": "Delta Air Lines",
        "text": "Delta Air Lines reported strong revenue growth in the airline sector. "
                "EBITDA margins improved significantly. The company operates a fleet "
                "of Boeing and Airbus aircraft.",
    },
    {
        "chunk_id": "chunk_002",
        "document_id": "doc_002",
        "doc_source": "transcript",
        "document_type": "Transcript",
        "document_date": datetime(2024, 6, 20),
        "company_name": "Delta Air Lines",
        "text": "During the earnings call, Delta management discussed capacity expansion "
                "and fuel hedging strategies for the upcoming quarters.",
    },
    {
        "chunk_id": "chunk_003",
        "document_id": "doc_003",
        "doc_source": "sec_filing",
        "document_type": "10-Q",
        "document_date": datetime(2024, 9, 1),
        "company_name": "Ford Motor",
        "text": "Ford Motor reported quarterly earnings with a focus on electric vehicle "
                "production. The F-150 Lightning saw increased deliveries. Capital "
                "expenditures rose due to EV factory investments.",
    },
    {
        "chunk_id": "chunk_004",
        "document_id": "doc_004",
        "doc_source": "sec_filing",
        "document_type": "10-K",
        "document_date": datetime(2023, 12, 31),
        "company_name": "Ford Motor",
        "text": "Risk factors include supply chain disruptions, semiconductor shortages, "
                "and competitive pressures in the autonomous vehicle market.",
    },
    {
        "chunk_id": "chunk_005",
        "document_id": "doc_005",
        "doc_source": "transcript",
        "document_type": "Transcript",
        "document_date": datetime(2024, 1, 15),
        "company_name": "JetBlue",
        "text": "JetBlue Airways discussed cost reduction initiatives and route "
                "optimization during the quarterly earnings conference call.",
    },
]


class TestBM25Scoring:
    """Test that BM25 returns results for known keyword queries."""

    def setup_method(self):
        self.retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=5)

    def test_keyword_query_returns_results(self):
        results = self.retriever.retrieve("Delta revenue growth airline")
        assert len(results) > 0

    def test_relevant_chunk_ranked_first(self):
        results = self.retriever.retrieve("Delta Air Lines revenue EBITDA")
        assert len(results) > 0
        # The first chunk should be the Delta 10-K chunk which has all these terms
        assert results[0]["chunk_id"] == "chunk_001"

    def test_ev_query_finds_ford(self):
        results = self.retriever.retrieve("electric vehicle EV production")
        assert len(results) > 0
        ford_ids = [r["chunk_id"] for r in results if r["company_name"] == "Ford Motor"]
        assert len(ford_ids) > 0, "Expected Ford Motor chunks for EV query"

    def test_earnings_call_query(self):
        results = self.retriever.retrieve("earnings call capacity fuel hedging")
        assert len(results) > 0
        # Delta transcript should rank high
        top_ids = [r["chunk_id"] for r in results[:2]]
        assert "chunk_002" in top_ids


class TestTopKLimit:
    """Test that top_k limit is respected."""

    def test_top_k_2(self):
        retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=2)
        results = retriever.retrieve("Delta revenue airline")
        assert len(results) <= 2

    def test_top_k_1(self):
        retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=1)
        results = retriever.retrieve("Delta revenue airline")
        assert len(results) <= 1

    def test_top_k_larger_than_corpus(self):
        retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=100)
        results = retriever.retrieve("Delta revenue airline")
        # Should return at most len(TEST_CHUNKS) results
        assert len(results) <= len(TEST_CHUNKS)


class TestMetadataFilters:
    """Test metadata filter application."""

    def setup_method(self):
        self.retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=10)

    def test_filter_by_company_name(self):
        results = self.retriever.retrieve(
            "revenue earnings", company_name="Ford Motor"
        )
        for r in results:
            assert r["company_name"] == "Ford Motor"

    def test_filter_by_doc_source(self):
        results = self.retriever.retrieve(
            "revenue earnings", doc_source="sec_filing"
        )
        for r in results:
            assert r["doc_source"] == "sec_filing"

    def test_filter_by_document_type(self):
        results = self.retriever.retrieve(
            "revenue earnings", document_type="Transcript"
        )
        for r in results:
            assert r["document_type"] == "Transcript"

    def test_combined_filters(self):
        results = self.retriever.retrieve(
            "revenue",
            company_name="Delta Air Lines",
            doc_source="sec_filing",
        )
        for r in results:
            assert r["company_name"] == "Delta Air Lines"
            assert r["doc_source"] == "sec_filing"

    def test_filter_excludes_nonmatching(self):
        results = self.retriever.retrieve(
            "Delta revenue airline", company_name="Ford Motor"
        )
        # Even though query is Delta-focused, filter should restrict to Ford
        for r in results:
            assert r["company_name"] == "Ford Motor"


class TestScoreField:
    """Test that results have the _score field."""

    def test_results_have_score(self):
        retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=5)
        results = retriever.retrieve("Delta revenue airline")
        for r in results:
            assert "_score" in r, "Result missing _score field"
            assert isinstance(r["_score"], float)
            assert r["_score"] > 0

    def test_scores_descending(self):
        retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=5)
        results = retriever.retrieve("Delta revenue airline")
        scores = [r["_score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Scores not in descending order"


class TestEmptyQuery:
    """Test behavior with empty or trivial queries."""

    def test_empty_query_string(self):
        retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=5)
        results = retriever.retrieve("")
        # Empty query may return no results (all scores 0)
        assert isinstance(results, list)

    def test_query_with_no_matching_terms(self):
        retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=5)
        results = retriever.retrieve("xyzzyplugh nonexistent")
        # Should return empty list since no terms match
        assert isinstance(results, list)


class TestDateFilter:
    """Test date range filtering."""

    def test_date_from_filter(self):
        retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=10)
        results = retriever.retrieve(
            "revenue earnings",
            date_from=datetime(2024, 6, 1),
        )
        for r in results:
            if r.get("document_date") is not None:
                assert r["document_date"] >= datetime(2024, 6, 1)

    def test_date_to_filter(self):
        retriever = BM25Retriever(chunks=TEST_CHUNKS, top_k=10)
        results = retriever.retrieve(
            "revenue earnings",
            date_to=datetime(2024, 3, 31),
        )
        for r in results:
            if r.get("document_date") is not None:
                assert r["document_date"] <= datetime(2024, 3, 31)
