"""Data models for the RAG evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalQuestion:
    """A test question for RAG evaluation."""
    id: str
    text: str
    category: str
    requires_simfin: bool = False
    requires_docs: bool = True
    expected_companies: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Configuration for a single RAG experiment."""
    name: str
    chunker: str             # "heading" | "token" | "sec_section" | "hybrid" | "recursive"
    retriever: str           # "dense" | "dense_mmr" | "bm25" | "hybrid_dense_bm25"
    embedding_model: str     # HuggingFace model name
    top_k: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 64
    multivector: bool = False  # build 3 specialized stores (transcript/sec_recursive/sec_section)
    hyde: bool = False         # enable query-adaptive HyDE in orchestrator

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chunker": self.chunker,
            "retriever": self.retriever,
            "embedding_model": self.embedding_model,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "multivector": self.multivector,
            "hyde": self.hyde,
        }

    def cache_key(self) -> str:
        """Unique key for chunk/index caching."""
        mv = "_mv" if self.multivector else ""
        return f"{self.chunker}_{self.chunk_size}_{self.chunk_overlap}_{self.embedding_model}{mv}"


@dataclass
class ExperimentResult:
    """Full output of running one question through one experiment config."""
    experiment_name: str
    question_id: str
    config: dict[str, Any]
    citations: list[dict[str, Any]]
    answer_text: str
    trace_events: list[dict[str, Any]]
    timing: dict[str, float]  # retrieval_ms, synthesis_ms, total_ms
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "question_id": self.question_id,
            "config": self.config,
            "citations": self.citations,
            "answer_text": self.answer_text,
            "trace_events": self.trace_events,
            "timing": self.timing,
            "timestamp": self.timestamp,
        }


@dataclass
class JudgeScore:
    """LLM judge evaluation scores for one experiment result."""
    experiment_name: str
    question_id: str
    context_relevance: int = 0       # 1-5
    context_precision: int = 0       # 1-5
    context_recall: int = 0          # 1-5
    faithfulness: int = 0            # 1-5
    answer_relevance: int = 0        # 1-5
    citation_accuracy: int = 0       # 1-5
    completeness: int = 0            # 1-5
    retrieval_latency_ms: float = 0.0
    reasoning: dict[str, str] = field(default_factory=dict)
    raw_judge_response: str = ""

    @property
    def composite_score(self) -> float:
        """Weighted average of all metrics."""
        scores = [
            self.context_relevance,
            self.context_precision,
            self.context_recall,
            self.faithfulness,
            self.answer_relevance,
            self.citation_accuracy,
            self.completeness,
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "question_id": self.question_id,
            "scores": {
                "context_relevance": self.context_relevance,
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "faithfulness": self.faithfulness,
                "answer_relevance": self.answer_relevance,
                "citation_accuracy": self.citation_accuracy,
                "completeness": self.completeness,
            },
            "composite_score": self.composite_score,
            "reasoning": self.reasoning,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "raw_judge_response": self.raw_judge_response,
        }


@dataclass
class ComparisonResult:
    """LLM judge comparative evaluation across experiments for one question."""
    question_id: str
    ranked_experiments: list[str] = field(default_factory=list)
    analysis: str = ""
    best_config: str = ""
    raw_judge_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "ranked_experiments": self.ranked_experiments,
            "analysis": self.analysis,
            "best_config": self.best_config,
            "raw_judge_response": self.raw_judge_response,
        }
