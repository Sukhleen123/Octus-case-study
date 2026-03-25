"""Tests for eval pipeline utilities: YAML loading, models, and judge parsing."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest
import yaml

from src.eval.judge import _parse_judge_response
from src.eval.models import (
    EvalQuestion,
    ExperimentConfig,
    ExperimentResult,
    JudgeScore,
)
from src.eval.run_rag_eval import _load_experiments, _load_questions


# ── Inline test data ──────────────────────────────────────────────────────────

SAMPLE_QUESTIONS_YAML = {
    "questions": [
        {
            "id": "q01",
            "text": "What is the revenue for Delta?",
            "category": "financial_data",
            "requires_simfin": True,
            "requires_docs": False,
            "expected_companies": ["Delta Air Lines"],
        },
        {
            "id": "q02",
            "text": "Summarize Ford 10-Q commentary.",
            "category": "sec_filing",
            "requires_simfin": False,
            "requires_docs": True,
            "expected_companies": ["Ford Motor"],
        },
        {
            "id": "q03",
            "text": "What are sector trends?",
            "category": "sector_overview",
        },
    ]
}

SAMPLE_EXPERIMENTS_YAML = {
    "experiments": [
        {
            "name": "chunker_heading",
            "chunker": "heading",
            "retriever": "dense",
            "embedding_model": "intfloat/e5-base-v2",
            "top_k": 10,
            "chunk_size": 512,
            "chunk_overlap": 64,
        },
        {
            "name": "retriever_bm25",
            "chunker": "token",
            "retriever": "bm25",
            "embedding_model": "intfloat/e5-base-v2",
        },
    ]
}


def _write_yaml(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


# ── Test loading eval questions from YAML ─────────────────────────────────────

class TestLoadQuestions:
    def test_loads_all_questions(self, tmp_path):
        yaml_path = str(tmp_path / "questions.yaml")
        _write_yaml(SAMPLE_QUESTIONS_YAML, yaml_path)

        questions = _load_questions(yaml_path)
        assert len(questions) == 3

    def test_question_fields(self, tmp_path):
        yaml_path = str(tmp_path / "questions.yaml")
        _write_yaml(SAMPLE_QUESTIONS_YAML, yaml_path)

        questions = _load_questions(yaml_path)
        q1 = questions[0]

        assert isinstance(q1, EvalQuestion)
        assert q1.id == "q01"
        assert q1.text == "What is the revenue for Delta?"
        assert q1.category == "financial_data"
        assert q1.requires_simfin is True
        assert q1.requires_docs is False
        assert q1.expected_companies == ["Delta Air Lines"]

    def test_question_defaults(self, tmp_path):
        yaml_path = str(tmp_path / "questions.yaml")
        _write_yaml(SAMPLE_QUESTIONS_YAML, yaml_path)

        questions = _load_questions(yaml_path)
        q3 = questions[2]

        # Missing fields should get defaults
        assert q3.requires_simfin is False
        assert q3.requires_docs is True
        assert q3.expected_companies == []

    def test_empty_questions_file(self, tmp_path):
        yaml_path = str(tmp_path / "questions.yaml")
        _write_yaml({"questions": []}, yaml_path)

        questions = _load_questions(yaml_path)
        assert questions == []


# ── Test loading experiment configs from YAML ─────────────────────────────────

class TestLoadExperiments:
    def test_loads_all_experiments(self, tmp_path):
        yaml_path = str(tmp_path / "grid.yaml")
        _write_yaml(SAMPLE_EXPERIMENTS_YAML, yaml_path)

        experiments = _load_experiments(yaml_path)
        assert len(experiments) == 2

    def test_experiment_fields(self, tmp_path):
        yaml_path = str(tmp_path / "grid.yaml")
        _write_yaml(SAMPLE_EXPERIMENTS_YAML, yaml_path)

        experiments = _load_experiments(yaml_path)
        e1 = experiments[0]

        assert isinstance(e1, ExperimentConfig)
        assert e1.name == "chunker_heading"
        assert e1.chunker == "heading"
        assert e1.retriever == "dense"
        assert e1.embedding_model == "intfloat/e5-base-v2"
        assert e1.top_k == 10
        assert e1.chunk_size == 512
        assert e1.chunk_overlap == 64

    def test_experiment_defaults(self, tmp_path):
        yaml_path = str(tmp_path / "grid.yaml")
        _write_yaml(SAMPLE_EXPERIMENTS_YAML, yaml_path)

        experiments = _load_experiments(yaml_path)
        e2 = experiments[1]

        # Missing top_k, chunk_size, chunk_overlap should get defaults
        assert e2.top_k == 10
        assert e2.chunk_size == 512
        assert e2.chunk_overlap == 64


# ── Test ExperimentConfig.to_dict() and cache_key() ───────────────────────────

class TestExperimentConfig:
    def test_to_dict(self):
        config = ExperimentConfig(
            name="test_exp",
            chunker="token",
            retriever="dense",
            embedding_model="intfloat/e5-base-v2",
            top_k=10,
            chunk_size=512,
            chunk_overlap=64,
        )
        d = config.to_dict()

        assert d["name"] == "test_exp"
        assert d["chunker"] == "token"
        assert d["retriever"] == "dense"
        assert d["embedding_model"] == "intfloat/e5-base-v2"
        assert d["top_k"] == 10
        assert d["chunk_size"] == 512
        assert d["chunk_overlap"] == 64

    def test_to_dict_keys(self):
        config = ExperimentConfig(
            name="test_exp",
            chunker="token",
            retriever="dense",
            embedding_model="intfloat/e5-base-v2",
        )
        d = config.to_dict()
        expected_keys = {
            "name", "chunker", "retriever", "embedding_model",
            "top_k", "chunk_size", "chunk_overlap",
        }
        assert set(d.keys()) == expected_keys

    def test_cache_key_format(self):
        config = ExperimentConfig(
            name="test_exp",
            chunker="token",
            retriever="dense",
            embedding_model="intfloat/e5-base-v2",
            chunk_size=512,
            chunk_overlap=64,
        )
        key = config.cache_key()
        assert key == "token_512_64_intfloat/e5-base-v2"

    def test_cache_key_varies_with_chunker(self):
        config_a = ExperimentConfig(
            name="a", chunker="token", retriever="dense",
            embedding_model="intfloat/e5-base-v2",
        )
        config_b = ExperimentConfig(
            name="b", chunker="heading", retriever="dense",
            embedding_model="intfloat/e5-base-v2",
        )
        assert config_a.cache_key() != config_b.cache_key()

    def test_cache_key_same_for_different_retriever(self):
        """cache_key does not include retriever, only chunker/size/model."""
        config_a = ExperimentConfig(
            name="a", chunker="token", retriever="dense",
            embedding_model="intfloat/e5-base-v2",
        )
        config_b = ExperimentConfig(
            name="b", chunker="token", retriever="bm25",
            embedding_model="intfloat/e5-base-v2",
        )
        assert config_a.cache_key() == config_b.cache_key()


# ── Test JudgeScore.composite_score ───────────────────────────────────────────

class TestJudgeScore:
    def test_composite_score_calculation(self):
        score = JudgeScore(
            experiment_name="test_exp",
            question_id="q01",
            context_relevance=5,
            context_precision=4,
            context_recall=3,
            faithfulness=5,
            answer_relevance=4,
            citation_accuracy=3,
            completeness=4,
        )
        expected = (5 + 4 + 3 + 5 + 4 + 3 + 4) / 7
        assert abs(score.composite_score - expected) < 1e-9

    def test_composite_score_all_zeros(self):
        score = JudgeScore(
            experiment_name="test_exp",
            question_id="q01",
        )
        assert score.composite_score == 0.0

    def test_composite_score_all_fives(self):
        score = JudgeScore(
            experiment_name="test_exp",
            question_id="q01",
            context_relevance=5,
            context_precision=5,
            context_recall=5,
            faithfulness=5,
            answer_relevance=5,
            citation_accuracy=5,
            completeness=5,
        )
        assert score.composite_score == 5.0

    def test_to_dict_includes_composite(self):
        score = JudgeScore(
            experiment_name="test_exp",
            question_id="q01",
            context_relevance=4,
            context_precision=3,
            context_recall=5,
            faithfulness=4,
            answer_relevance=3,
            citation_accuracy=5,
            completeness=4,
        )
        d = score.to_dict()
        assert "composite_score" in d
        assert abs(d["composite_score"] - score.composite_score) < 1e-9

    def test_to_dict_structure(self):
        score = JudgeScore(
            experiment_name="test_exp",
            question_id="q01",
            context_relevance=4,
            reasoning={"context_relevance": "Good chunks selected"},
            raw_judge_response="raw json here",
        )
        d = score.to_dict()
        assert d["experiment_name"] == "test_exp"
        assert d["question_id"] == "q01"
        assert "scores" in d
        assert d["scores"]["context_relevance"] == 4
        assert d["reasoning"]["context_relevance"] == "Good chunks selected"
        assert d["raw_judge_response"] == "raw json here"


# ── Test _parse_judge_response ────────────────────────────────────────────────

class TestParseJudgeResponse:
    def test_plain_json(self):
        text = json.dumps({
            "context_relevance": {"reasoning": "Good", "score": 4},
            "faithfulness": {"reasoning": "Grounded", "score": 5},
        })
        data = _parse_judge_response(text)
        assert data["context_relevance"]["score"] == 4
        assert data["faithfulness"]["score"] == 5

    def test_json_with_markdown_code_fence(self):
        text = '```json\n{"context_relevance": {"reasoning": "Ok", "score": 3}}\n```'
        data = _parse_judge_response(text)
        assert data["context_relevance"]["score"] == 3

    def test_json_with_plain_code_fence(self):
        text = '```\n{"context_relevance": {"reasoning": "Ok", "score": 3}}\n```'
        data = _parse_judge_response(text)
        assert data["context_relevance"]["score"] == 3

    def test_json_with_surrounding_whitespace(self):
        text = '\n\n  {"key": "value"}  \n\n'
        data = _parse_judge_response(text)
        assert data["key"] == "value"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_judge_response("this is not json at all")

    def test_full_judge_response(self):
        response = json.dumps({
            "context_relevance": {"reasoning": "All chunks relevant", "score": 5},
            "context_precision": {"reasoning": "Top ranked well", "score": 4},
            "context_recall": {"reasoning": "Key info present", "score": 4},
            "faithfulness": {"reasoning": "Well grounded", "score": 5},
            "answer_relevance": {"reasoning": "Directly answers", "score": 5},
            "citation_accuracy": {"reasoning": "Correct refs", "score": 4},
            "completeness": {"reasoning": "Covers all aspects", "score": 4},
        })
        wrapped = f"```json\n{response}\n```"
        data = _parse_judge_response(wrapped)
        assert len(data) == 7
        for metric in data.values():
            assert "score" in metric
            assert "reasoning" in metric


# ── Test ExperimentResult.to_dict() ───────────────────────────────────────────

class TestExperimentResult:
    def test_to_dict_serialization(self):
        result = ExperimentResult(
            experiment_name="test_exp",
            question_id="q01",
            config={"chunker": "token", "retriever": "dense"},
            citations=[
                {"ref_number": 1, "cited_text": "Some citation"}
            ],
            answer_text="The revenue was $500M.",
            trace_events=[
                {"event_type": "retrieval", "payload": {}}
            ],
            timing={"total_ms": 123.4, "retrieval_ms": 50.0},
            timestamp="2024-06-15T12:00:00Z",
        )
        d = result.to_dict()

        assert d["experiment_name"] == "test_exp"
        assert d["question_id"] == "q01"
        assert d["config"]["chunker"] == "token"
        assert len(d["citations"]) == 1
        assert d["answer_text"] == "The revenue was $500M."
        assert len(d["trace_events"]) == 1
        assert d["timing"]["total_ms"] == 123.4
        assert d["timestamp"] == "2024-06-15T12:00:00Z"

    def test_to_dict_keys(self):
        result = ExperimentResult(
            experiment_name="test_exp",
            question_id="q01",
            config={},
            citations=[],
            answer_text="Answer",
            trace_events=[],
            timing={},
        )
        d = result.to_dict()
        expected_keys = {
            "experiment_name", "question_id", "config",
            "citations", "answer_text",
            "trace_events", "timing", "timestamp",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_with_empty_fields(self):
        result = ExperimentResult(
            experiment_name="",
            question_id="",
            config={},
            citations=[],
            answer_text="",
            trace_events=[],
            timing={},
            timestamp="",
        )
        d = result.to_dict()
        assert d["experiment_name"] == ""
        assert d["timing"] == {}
