"""Read/write eval results to structured directory layout."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.eval.models import ComparisonResult, ExperimentResult, JudgeScore

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path("data/eval")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ── ExperimentResult ──────────────────────────────────────────────────────────

def save_experiment_result(result: ExperimentResult, output_dir: Path = _DEFAULT_OUTPUT_DIR) -> Path:
    """Save one experiment result as JSON."""
    dir_ = output_dir / "results" / result.experiment_name
    _ensure_dir(dir_)
    path = dir_ / f"{result.question_id}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_experiment_results(output_dir: Path = _DEFAULT_OUTPUT_DIR) -> list[ExperimentResult]:
    """Load all experiment results from the output directory."""
    results_dir = output_dir / "results"
    if not results_dir.exists():
        return []
    results = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for f in sorted(exp_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                results.append(ExperimentResult(**data))
            except Exception as e:
                logger.warning("Failed to load result %s: %s", f, e)
    return results


def load_results_for_experiment(
    experiment_name: str,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> list[ExperimentResult]:
    """Load results for a specific experiment."""
    dir_ = output_dir / "results" / experiment_name
    if not dir_.exists():
        return []
    results = []
    for f in sorted(dir_.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append(ExperimentResult(**data))
        except Exception as e:
            logger.warning("Failed to load result %s: %s", f, e)
    return results


# ── JudgeScore ────────────────────────────────────────────────────────────────

def save_judge_score(score: JudgeScore, output_dir: Path = _DEFAULT_OUTPUT_DIR) -> Path:
    """Save one judge score as JSON."""
    dir_ = output_dir / "scores" / score.experiment_name
    _ensure_dir(dir_)
    path = dir_ / f"{score.question_id}_score.json"
    path.write_text(json.dumps(score.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_judge_scores(output_dir: Path = _DEFAULT_OUTPUT_DIR) -> list[JudgeScore]:
    """Load all judge scores from the output directory."""
    scores_dir = output_dir / "scores"
    if not scores_dir.exists():
        return []
    scores = []
    for exp_dir in sorted(scores_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for f in sorted(exp_dir.glob("*_score.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                scores.append(JudgeScore(
                    experiment_name=data["experiment_name"],
                    question_id=data["question_id"],
                    context_relevance=data["scores"]["context_relevance"],
                    context_precision=data["scores"]["context_precision"],
                    context_recall=data["scores"]["context_recall"],
                    faithfulness=data["scores"]["faithfulness"],
                    answer_relevance=data["scores"]["answer_relevance"],
                    citation_accuracy=data["scores"]["citation_accuracy"],
                    completeness=data["scores"]["completeness"],
                    retrieval_latency_ms=data.get("retrieval_latency_ms", 0.0),
                    reasoning=data.get("reasoning", {}),
                    raw_judge_response=data.get("raw_judge_response", ""),
                ))
            except Exception as e:
                logger.warning("Failed to load score %s: %s", f, e)
    return scores


# ── ComparisonResult ──────────────────────────────────────────────────────────

def save_comparison(comp: ComparisonResult, output_dir: Path = _DEFAULT_OUTPUT_DIR) -> Path:
    """Save one comparison result as JSON."""
    dir_ = output_dir / "comparisons"
    _ensure_dir(dir_)
    path = dir_ / f"{comp.question_id}_comparison.json"
    path.write_text(json.dumps(comp.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_comparisons(output_dir: Path = _DEFAULT_OUTPUT_DIR) -> list[ComparisonResult]:
    """Load all comparison results."""
    dir_ = output_dir / "comparisons"
    if not dir_.exists():
        return []
    comps = []
    for f in sorted(dir_.glob("*_comparison.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            comps.append(ComparisonResult(**data))
        except Exception as e:
            logger.warning("Failed to load comparison %s: %s", f, e)
    return comps
