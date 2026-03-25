"""
RAG Evaluation Pipeline CLI.

Usage:
    # Step 1: build all vector stores (GPU embedding, Pinecone upsert) — no LLM calls
    python -m src.eval.run_rag_eval --build-indexes-only

    # Step 2: generate answers (indexes load from cache, Claude Haiku synthesis)
    python -m src.eval.run_rag_eval --num-questions 10 --skip-existing

    # Step 3: resume if rate-limited mid-run
    python -m src.eval.run_rag_eval --num-questions 10 --skip-existing

    # Step 4: run judge + report on collected results
    python -m src.eval.run_rag_eval --judge-only --compare

    # Run all experiments on all questions with judge scoring + comparison
    python -m src.eval.run_rag_eval --experiments all --questions all --judge --compare

    # Run specific experiments
    python -m src.eval.run_rag_eval --experiments chunker_heading retriever_bm25

    # Specify output directory
    python -m src.eval.run_rag_eval --output-dir data/eval_v2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.app.settings import Settings
from src.eval.models import ComparisonResult, EvalQuestion, ExperimentConfig, JudgeScore

logger = logging.getLogger(__name__)


def _load_questions(path: str = "configs/eval_questions.yaml") -> list[EvalQuestion]:
    """Load evaluation questions from YAML."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        EvalQuestion(
            id=q["id"],
            text=q["text"],
            category=q.get("category", ""),
            requires_simfin=q.get("requires_simfin", False),
            requires_docs=q.get("requires_docs", True),
            expected_companies=q.get("expected_companies", []),
        )
        for q in data.get("questions", [])
    ]


def _load_experiments(path: str = "configs/eval_grid.yaml") -> list[ExperimentConfig]:
    """Load experiment configurations from YAML."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        ExperimentConfig(
            name=e["name"],
            chunker=e["chunker"],
            retriever=e["retriever"],
            embedding_model=e["embedding_model"],
            top_k=e.get("top_k", 10),
            chunk_size=e.get("chunk_size", 512),
            chunk_overlap=e.get("chunk_overlap", 64),
            multivector=e.get("multivector", False),
            hyde=e.get("hyde", False),
        )
        for e in data.get("experiments", [])
    ]


def _make_llm_client(settings: Settings):
    """Create Anthropic client for judge evaluations."""
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument(
        "--experiments", nargs="*", default=["all"],
        help="Experiment names to run (or 'all')",
    )
    parser.add_argument(
        "--questions", nargs="*", default=["all"],
        help="Question IDs to evaluate (or 'all')",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run LLM judge on experiment results",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run comparative judge across experiments per question",
    )
    parser.add_argument(
        "--judge-only", action="store_true",
        help="Skip experiments, only run judge on existing results",
    )
    parser.add_argument(
        "--output-dir", default="data/eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--questions-file", default="configs/eval_questions.yaml",
        help="Path to questions YAML",
    )
    parser.add_argument(
        "--grid-file", default="configs/eval_grid.yaml",
        help="Path to experiment grid YAML",
    )
    parser.add_argument(
        "--num-questions", type=int, default=0,
        help="Randomly sample N questions from the pool (0 = use all/specified)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for --num-questions sampling (default: 42)",
    )
    parser.add_argument(
        "--build-indexes-only", action="store_true",
        help=(
            "Build all vector stores (production Pinecone + eval FAISS indexes) "
            "without running any questions or LLM calls. "
            "Run this first on GPU, then re-run without this flag to generate answers."
        ),
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help=(
            "Skip questions that already have a result JSON in the output directory. "
            "Allows resuming a partial run after a rate-limit interruption."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    settings = Settings()

    # Load questions and experiments
    all_questions = _load_questions(args.questions_file)
    all_experiments = _load_experiments(args.grid_file)

    # Filter
    if "all" not in args.questions:
        all_questions = [q for q in all_questions if q.id in args.questions]
    if "all" not in args.experiments:
        all_experiments = [e for e in all_experiments if e.name in args.experiments]

    # Random sampling
    if args.num_questions > 0 and args.num_questions < len(all_questions):
        import random
        rng = random.Random(args.seed)
        all_questions = rng.sample(all_questions, args.num_questions)
        logger.info("Sampled %d random questions (seed=%d)", args.num_questions, args.seed)

    logger.info("Questions: %d, Experiments: %d", len(all_questions), len(all_experiments))

    # ── Phase 0: Build indexes only (no LLM calls) ──────────────────────────
    if args.build_indexes_only:
        from src.eval.pipeline import _cache_hash, build_eval_pipeline

        logger.info("Connecting to pre-built indexes via loader...")
        from src.app.loader import load_context
        load_context(settings)
        logger.info("Index connection complete.")

        # Pre-build eval FAISS indexes; deduplicate by cache hash so experiments
        # sharing the same (chunker, size, overlap, embedding) only embed once.
        seen_hashes: set[str] = set()
        for config in all_experiments:
            h = _cache_hash(config)
            if h in seen_hashes:
                logger.info(
                    "Skipping duplicate index for %s (hash %s already built)", config.name, h
                )
                continue
            seen_hashes.add(h)
            logger.info("Pre-building eval FAISS index for: %s (hash %s)", config.name, h)
            build_eval_pipeline(config, settings)

        logger.info(
            "All vector stores built. Re-run without --build-indexes-only to generate answers."
        )
        return

    # ── Phase 1: Run experiments ────────────────────────────────────────────
    from src.eval.pipeline import run_single_experiment
    from src.eval.storage import (
        load_experiment_results,
        load_judge_scores,
        save_comparison,
        save_experiment_result,
        save_judge_score,
    )

    if not args.judge_only:
        for config in all_experiments:
            logger.info("Running experiment: %s", config.name)

            questions_to_run = all_questions
            if args.skip_existing:
                results_dir = output_dir / "results" / config.name
                done = (
                    {f.stem for f in results_dir.glob("*.json")}
                    if results_dir.exists()
                    else set()
                )
                questions_to_run = [q for q in all_questions if q.id not in done]
                if not questions_to_run:
                    logger.info("All questions already done for %s, skipping.", config.name)
                    continue
                if done:
                    logger.info(
                        "Skipping %d already-done questions for %s; running %d remaining.",
                        len(done), config.name, len(questions_to_run),
                    )

            results = run_single_experiment(config, questions_to_run, settings)
            for r in results:
                save_experiment_result(r, output_dir)
            logger.info("Saved %d results for %s", len(results), config.name)

    # ── Phase 2: Judge evaluation ───────────────────────────────────────────
    if args.judge or args.judge_only:
        from src.eval.judge import judge_comparative, judge_single

        llm_client = _make_llm_client(settings)
        if llm_client is None:
            logger.error("LLM client required for judge. Set ANTHROPIC_API_KEY.")
            sys.exit(1)

        all_results = load_experiment_results(output_dir)
        logger.info("Loaded %d experiment results for judging", len(all_results))

        # Filter to requested experiments/questions
        if "all" not in args.experiments:
            all_results = [r for r in all_results if r.experiment_name in args.experiments]
        if "all" not in args.questions:
            all_results = [r for r in all_results if r.question_id in [q.id for q in all_questions]]

        # Single evaluation
        judge_scores: list[JudgeScore] = []
        for result in all_results:
            question = next((q for q in all_questions if q.id == result.question_id), None)
            if question is None:
                continue
            logger.info("Judging %s / %s", result.experiment_name, result.question_id)
            score = judge_single(question, result, llm_client, model=settings.llm_model)
            save_judge_score(score, output_dir)
            judge_scores.append(score)
            logger.info("  Composite: %.2f", score.composite_score)

        # Comparative evaluation
        comparisons: list[ComparisonResult] = []
        if args.compare:
            from collections import defaultdict
            by_question: dict[str, list[ExperimentResult]] = defaultdict(list)
            for r in all_results:
                by_question[r.question_id].append(r)

            for qid, q_results in by_question.items():
                if len(q_results) < 2:
                    continue
                question = next((q for q in all_questions if q.id == qid), None)
                if question is None:
                    continue
                logger.info("Comparing %d configs for %s", len(q_results), qid)
                comp = judge_comparative(question, q_results, llm_client, model=settings.llm_model)
                save_comparison(comp, output_dir)
                comparisons.append(comp)
                logger.info("  Best: %s", comp.best_config)
    else:
        judge_scores = load_judge_scores(output_dir)
        comparisons = None

    # ── Phase 3: Generate report ────────────────────────────────────────────
    if judge_scores:
        from src.eval.report import generate_summary_report
        report = generate_summary_report(judge_scores, comparisons, output_dir)
        logger.info("Report generated:\n%s", report[:500])
    else:
        logger.info("No judge scores to report.")

    logger.info("Evaluation complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
