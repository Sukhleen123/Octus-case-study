"""
LLM-as-Judge: evaluates RAG pipeline outputs using Claude.

Provides two evaluation modes:
  1. Single evaluation: score one experiment result on 7 metrics
  2. Comparative evaluation: rank multiple experiment results for the same question
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.eval.models import (
    ComparisonResult,
    EvalQuestion,
    ExperimentResult,
    JudgeScore,
)

logger = logging.getLogger(__name__)

_SINGLE_EVAL_PROMPT = """\
You are an expert evaluator of Retrieval-Augmented Generation (RAG) systems \
for financial document analysis. You will evaluate a RAG system's response to \
a financial question.

## How this RAG system works

This is a multi-agent system with two retrieval agents:

1. **DocAgent** — performs semantic search over an Octus document corpus \
(SEC filings, earnings call transcripts) using a FAISS vector index. \
It retrieves the top-{top_k} most similar chunks for the query.

2. **SimFinAgent** — fetches structured financial metrics (Revenue, Net Income, \
EPS, etc.) from batch-ingested SimFin data files for each relevant company ticker.

After both agents run, a **SynthesisAgent** receives ALL retrieved material and \
calls Claude to generate the final answer. Claude only includes reference numbers \
(e.g. [1], [5]) for the sources it actually uses. Unused retrieved material is \
silently dropped from the final citation list.

## What you are evaluating

- **Retrieved Doc Chunks** — ALL {num_chunks} document chunks the DocAgent \
fetched from the vector index, ranked by similarity score. This includes chunks \
that were NOT cited in the final answer. Use this to judge context relevance, \
precision, and recall.

- **Retrieved SimFin Rows** — ALL {num_simfin_rows} financial metric rows the \
SimFinAgent fetched. This includes rows that were NOT cited. Use this to assess \
whether the necessary financial data was available.

- **Final Citations** — ONLY the sources Claude actually referenced in its \
answer (ref_number > 0). Comparing this against the retrieved sets shows which \
retrieved material was useful vs ignored.

Score each metric on a 1-5 scale where 1=Poor, 2=Below Average, 3=Adequate, \
4=Good, 5=Excellent.

IMPORTANT: For each metric, first provide your chain-of-thought reasoning, \
then assign the score. Be specific about what the response did well or poorly.

## Question
{question}

## Retrieved Doc Chunks ({num_chunks} chunks, all fetched — including uncited)
{context}

## Retrieved SimFin Financial Rows ({num_simfin_rows} rows, all fetched — including uncited)
{simfin_rows}

## Generated Answer
{answer}

## Final Citations (only sources Claude actually referenced)
{citations}

## Scoring Rubric

### 1. Context Relevance (1-5)
Are the retrieved chunks pertinent to the query?
- 5: All chunks are directly relevant to answering the query
- 4: Most chunks are relevant with minor tangential content
- 3: Mix of relevant and tangentially related chunks
- 2: Few chunks are directly relevant
- 1: Most chunks are unrelated to the query

### 2. Context Precision (1-5)
Are the most relevant chunks ranked highest (appearing first)?
- 5: The top 3 chunks are the most relevant in the entire set
- 4: Most high-relevance chunks appear in top positions
- 3: Relevant chunks are scattered throughout the ranking
- 2: Important chunks appear mostly in lower positions
- 1: The most relevant chunks appear near the bottom

### 3. Context Recall (1-5)
Was all information needed to answer the question successfully retrieved?
- 5: All necessary information is present in the retrieved context
- 4: Nearly all necessary information is present
- 3: Some important information is missing but the core is present
- 2: Significant information gaps that affect answer quality
- 1: Critical information needed to answer is missing

### 4. Faithfulness / Grounding (1-5)
Is the answer derived solely from the retrieved context, without hallucination?
- 5: Every claim is directly supported by the retrieved context
- 4: Nearly all claims are supported with minimal unsupported inferences
- 3: Most claims are supported but some minor unsupported assertions exist
- 2: Several claims lack support from the context
- 1: The answer contains significant claims not found in the context

### 5. Answer Relevance (1-5)
Does the answer directly and completely address the question asked?
- 5: The answer fully addresses the question with appropriate detail
- 4: The answer addresses the main question with minor gaps
- 3: The answer partially addresses the question
- 2: The answer only tangentially relates to the question
- 1: The answer does not address the question

### 6. Citation Accuracy (1-5)
Are citations correctly attributed and traceable to specific sources?
- 5: All cited claims have correct, traceable [N] references to specific sources
- 4: Most citations are correct with minor attribution issues
- 3: Some citations are present but inconsistently applied
- 2: Citations are sparse or frequently incorrect
- 1: Citations are missing, wrong, or untraceable

### 7. Completeness (1-5)
Does the answer cover all aspects of the question?
- 5: All aspects of the question are thoroughly addressed
- 4: Most aspects are addressed with minor omissions
- 3: The main point is addressed but some aspects are missing
- 2: Only a few aspects of the question are addressed
- 1: The answer only covers a small portion of what was asked

Respond with ONLY valid JSON in exactly this format (no markdown code fences):
{{
  "context_relevance": {{"reasoning": "...", "score": N}},
  "context_precision": {{"reasoning": "...", "score": N}},
  "context_recall": {{"reasoning": "...", "score": N}},
  "faithfulness": {{"reasoning": "...", "score": N}},
  "answer_relevance": {{"reasoning": "...", "score": N}},
  "citation_accuracy": {{"reasoning": "...", "score": N}},
  "completeness": {{"reasoning": "...", "score": N}}
}}"""

_COMPARATIVE_PROMPT = """\
You are comparing {num_configs} different RAG pipeline configurations that \
answered the same financial question. Rank them from best to worst and \
explain your reasoning.

## Question
{question}

{configs_block}

Consider these factors when ranking:
1. Answer quality, accuracy, and completeness
2. Citation accuracy and traceability
3. Whether the retrieved context was sufficient and relevant
4. Overall coherence and usefulness for a financial analyst
5. Retrieval latency (lower is better, but quality matters more)

Respond with ONLY valid JSON (no markdown code fences):
{{
  "ranking": ["config_name_best", "config_name_second", ...],
  "analysis": "Detailed comparative analysis explaining why...",
  "best_config": "config_name_best",
  "best_config_strengths": "Why this config produced the best result..."
}}"""


def _extract_doc_chunks(trace_events: list[dict]) -> list[dict]:
    """Extract all retrieved doc chunks from the retrieval_results trace event."""
    for event in trace_events:
        if event.get("event_type") == "retrieval_results":
            return event.get("payload", {}).get("chunks", [])
    return []


def _extract_simfin_rows(trace_events: list[dict]) -> list[dict]:
    """Extract all retrieved SimFin rows from the simfin_results trace event."""
    for event in trace_events:
        if event.get("event_type") == "simfin_results":
            return event.get("payload", {}).get("rows", [])
    return []


def _format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks for the judge prompt."""
    if not chunks:
        return "(No document chunks retrieved — DocAgent either did not run or returned nothing)"
    lines = []
    for i, c in enumerate(chunks, 1):
        source = c.get("doc_source", "unknown")
        company = c.get("company_name", "unknown")
        doc_type = c.get("document_type", "")
        date = str(c.get("document_date", ""))[:10]
        section = c.get("section_title", "")
        text = c.get("text", "")[:500]
        score = c.get("_score", 0.0)
        header = f"[Chunk {i}] {source.upper()} / {doc_type} | {company} | {date}"
        if section:
            header += f" | Section: {section}"
        header += f" | Similarity: {score:.3f}"
        lines.append(f"{header}\n{text}\n")
    return "\n".join(lines)


def _format_simfin_rows(rows: list[dict]) -> str:
    """Format retrieved SimFin rows for the judge prompt."""
    if not rows:
        return "(No SimFin rows retrieved — SimFinAgent either did not run or found no data)"
    lines = []
    for r in rows:
        ticker = r.get("ticker", "")
        period = r.get("fiscal_period", "")
        year = r.get("fiscal_year", "")
        stmt = r.get("statement_type", "")
        metric = r.get("metric_name", "")
        value = r.get("metric_value", "")
        unit = r.get("metric_unit", "")
        lines.append(f"  {ticker} {period} FY{year} [{stmt}] {metric}: {value} ({unit})")
    return "\n".join(lines)


def _format_citations(citations: list[dict]) -> str:
    """Format citations for the judge prompt."""
    if not citations:
        return "(No citations)"
    lines = []
    for c in citations:
        ref = c.get("ref_number", 0)
        if "ticker" in c:
            lines.append(f"[{ref}] SimFin: {c.get('ticker')} {c.get('fiscal_period')} "
                         f"FY{c.get('fiscal_year')} - {c.get('metric_name')}: "
                         f"{c.get('metric_value', '')} {c.get('metric_unit', '')}")
        else:
            doc_source = c.get("doc_source", "")
            doc_type = c.get("document_type", "")
            doc_date = str(c.get("document_date", ""))[:10]
            text = c.get("cited_text", "")[:200]
            lines.append(f"[{ref}] {doc_source.upper()} / {doc_type} ({doc_date}): {text}")
    return "\n".join(lines)


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Parse JSON from judge response, handling markdown code fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def judge_single(
    question: EvalQuestion,
    result: ExperimentResult,
    llm_client: Any,
    model: str = "claude-sonnet-4-6",
) -> JudgeScore:
    """
    Evaluate one experiment result with the LLM judge.

    Returns a JudgeScore with 7 metric scores and reasoning.
    """
    doc_chunks = _extract_doc_chunks(result.trace_events)
    simfin_rows = _extract_simfin_rows(result.trace_events)

    prompt = _SINGLE_EVAL_PROMPT.format(
        question=question.text,
        top_k=result.config.get("top_k", 10),
        num_chunks=len(doc_chunks),
        context=_format_chunks(doc_chunks),
        num_simfin_rows=len(simfin_rows),
        simfin_rows=_format_simfin_rows(simfin_rows),
        answer=result.answer_text,
        citations=_format_citations(result.citations),
    )

    try:
        response = llm_client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
        data = _parse_judge_response(raw_text)

        reasoning = {}
        scores = {}
        for metric in [
            "context_relevance", "context_precision", "context_recall",
            "faithfulness", "answer_relevance", "citation_accuracy", "completeness",
        ]:
            entry = data.get(metric, {})
            scores[metric] = int(entry.get("score", 0))
            reasoning[metric] = entry.get("reasoning", "")

        return JudgeScore(
            experiment_name=result.experiment_name,
            question_id=result.question_id,
            context_relevance=scores.get("context_relevance", 0),
            context_precision=scores.get("context_precision", 0),
            context_recall=scores.get("context_recall", 0),
            faithfulness=scores.get("faithfulness", 0),
            answer_relevance=scores.get("answer_relevance", 0),
            citation_accuracy=scores.get("citation_accuracy", 0),
            completeness=scores.get("completeness", 0),
            retrieval_latency_ms=result.timing.get("total_ms", 0.0),
            reasoning=reasoning,
            raw_judge_response=raw_text,
        )
    except Exception as e:
        logger.error("Judge evaluation failed for %s/%s: %s",
                     result.experiment_name, result.question_id, e)
        return JudgeScore(
            experiment_name=result.experiment_name,
            question_id=result.question_id,
            raw_judge_response=f"ERROR: {e}",
        )


def judge_comparative(
    question: EvalQuestion,
    results: list[ExperimentResult],
    llm_client: Any,
    model: str = "claude-sonnet-4-6",
) -> ComparisonResult:
    """
    Compare multiple experiment results for the same question.

    Returns a ComparisonResult with rankings and analysis.
    """
    configs_parts = []
    label_to_name: dict[str, str] = {}
    for i, r in enumerate(results):
        label = chr(65 + i)  # A, B, C, ...
        label_to_name[label] = r.experiment_name
        doc_chunks = _extract_doc_chunks(r.trace_events)
        simfin_rows = _extract_simfin_rows(r.trace_events)
        block = (
            f"## Configuration {label}: {r.experiment_name}\n"
            f"Pipeline: chunker={r.config.get('chunker')}, "
            f"retriever={r.config.get('retriever')}, "
            f"embedding={r.config.get('embedding_model')}, "
            f"top_k={r.config.get('top_k')}\n"
            f"Doc chunks retrieved: {len(doc_chunks)} | "
            f"SimFin rows retrieved: {len(simfin_rows)} | "
            f"Final citations used: {len(r.citations)}\n"
            f"Latency: {r.timing.get('total_ms', 0):.0f}ms\n"
            f"Answer:\n{r.answer_text[:1500]}\n"
        )
        configs_parts.append(block)

    def _translate_label(s: str) -> str:
        """Translate 'Configuration A' or 'A' back to the experiment name."""
        s = s.strip()
        if s in label_to_name:
            return label_to_name[s]
        if s.startswith("Configuration ") and s[-1] in label_to_name:
            return label_to_name[s[-1]]
        # Last-resort: any single trailing letter
        if s and s[-1] in label_to_name:
            return label_to_name[s[-1]]
        return s

    prompt = _COMPARATIVE_PROMPT.format(
        num_configs=len(results),
        question=question.text,
        configs_block="\n".join(configs_parts),
    )

    try:
        response = llm_client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
        data = _parse_judge_response(raw_text)

        return ComparisonResult(
            question_id=question.id,
            ranked_experiments=[_translate_label(r) for r in data.get("ranking", [])],
            analysis=data.get("analysis", ""),
            best_config=_translate_label(data.get("best_config", "")),
            raw_judge_response=raw_text,
        )
    except Exception as e:
        logger.error("Comparative judge failed for %s: %s", question.id, e)
        return ComparisonResult(
            question_id=question.id,
            raw_judge_response=f"ERROR: {e}",
        )
