"""Generate summary reports from RAG evaluation results."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.eval.models import ComparisonResult, JudgeScore

logger = logging.getLogger(__name__)

_METRICS = [
    "context_relevance", "context_precision", "context_recall",
    "faithfulness", "answer_relevance", "citation_accuracy", "completeness",
]


def generate_summary_report(
    scores: list[JudgeScore],
    comparisons: list[ComparisonResult] | None = None,
    output_dir: Path = Path("data/eval"),
) -> str:
    """
    Generate a markdown evaluation report.

    Returns the report as a string and writes it to output_dir/eval_report.md.
    """
    if not scores:
        return "_No evaluation scores found._"

    # Aggregate scores by experiment
    by_experiment: dict[str, list[JudgeScore]] = defaultdict(list)
    for s in scores:
        by_experiment[s.experiment_name].append(s)

    # Compute averages per experiment
    leaderboard: list[dict[str, Any]] = []
    for exp_name, exp_scores in by_experiment.items():
        n = len(exp_scores)
        avg = {m: 0.0 for m in _METRICS}
        avg_latency = 0.0
        for s in exp_scores:
            for m in _METRICS:
                avg[m] += getattr(s, m, 0) / n
            avg_latency += s.retrieval_latency_ms / n

        composite = sum(avg.values()) / len(avg)
        leaderboard.append({
            "experiment": exp_name,
            "composite": round(composite, 2),
            "n_questions": n,
            "avg_latency_ms": round(avg_latency, 0),
            **{m: round(v, 2) for m, v in avg.items()},
        })

    leaderboard.sort(key=lambda x: x["composite"], reverse=True)

    # Build report
    lines = ["# RAG Evaluation Report\n"]

    # Leaderboard
    lines.append("## Leaderboard\n")
    lines.append("| Rank | Experiment | Composite | Ctx Rel | Ctx Prec | Ctx Rec | "
                 "Faith | Ans Rel | Cite Acc | Complete | Latency (ms) | N |")
    lines.append("|------|-----------|-----------|---------|----------|---------|"
                 "-------|---------|----------|----------|--------------|---|")
    for i, row in enumerate(leaderboard, 1):
        lines.append(
            f"| {i} | {row['experiment']} | **{row['composite']}** | "
            f"{row['context_relevance']} | {row['context_precision']} | "
            f"{row['context_recall']} | {row['faithfulness']} | "
            f"{row['answer_relevance']} | {row['citation_accuracy']} | "
            f"{row['completeness']} | {row['avg_latency_ms']:.0f} | "
            f"{row['n_questions']} |"
        )
    lines.append("")

    # Best config per metric
    lines.append("## Best Configuration Per Metric\n")
    for m in _METRICS:
        best = max(leaderboard, key=lambda x: x[m])
        label = m.replace("_", " ").title()
        lines.append(f"- **{label}**: {best['experiment']} ({best[m]})")
    lines.append("")

    # Per-question breakdown
    lines.append("## Per-Question Analysis\n")
    by_question: dict[str, list[JudgeScore]] = defaultdict(list)
    for s in scores:
        by_question[s.question_id].append(s)

    for qid in sorted(by_question.keys()):
        q_scores = by_question[qid]
        best = max(q_scores, key=lambda s: s.composite_score)
        worst = min(q_scores, key=lambda s: s.composite_score)
        lines.append(f"### {qid}")
        lines.append(f"- Best: **{best.experiment_name}** (composite: {best.composite_score:.2f})")
        lines.append(f"- Worst: {worst.experiment_name} (composite: {worst.composite_score:.2f})")
        lines.append("")

    # Comparison results
    if comparisons:
        lines.append("## Head-to-Head Comparison Results\n")
        win_count: dict[str, int] = defaultdict(int)
        for comp in comparisons:
            if comp.best_config:
                win_count[comp.best_config] += 1

        if win_count:
            lines.append("| Config | #1 Rankings |")
            lines.append("|--------|-------------|")
            for config, count in sorted(win_count.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {config} | {count} |")
            lines.append("")

        for comp in comparisons:
            lines.append(f"### {comp.question_id}")
            if comp.ranked_experiments:
                lines.append(f"Ranking: {' > '.join(comp.ranked_experiments)}")
            if comp.analysis:
                lines.append(f"\n{comp.analysis[:500]}")
            lines.append("")

    report = "\n".join(lines)

    # Write files
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eval_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", report_path)

    # Machine-readable summary
    summary = {
        "leaderboard": leaderboard,
        "total_experiments": len(by_experiment),
        "total_questions_evaluated": len(by_question),
        "total_scores": len(scores),
    }
    summary_path = output_dir / "eval_results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return report
