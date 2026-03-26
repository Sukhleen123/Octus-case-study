"""Human-readable citation formatting."""

from __future__ import annotations

from collections import defaultdict

from src.citations.models import OctusCitation, SimFinCitation


def format_citations_from_dicts(citations: list[dict]) -> str:
    """
    Format a list of serialized citation dicts (from SynthesisOutput.citations)
    by reconstructing them into typed objects and delegating to format_citations_block.
    """
    objects = []
    for d in citations:
        if "ticker" in d:
            objects.append(SimFinCitation(**{k: v for k, v in d.items() if k != "type"}))
        else:
            objects.append(OctusCitation(**{k: v for k, v in d.items() if k != "type"}))
    return format_citations_block(objects)


def format_citation(citation: OctusCitation | SimFinCitation) -> str:
    """Return a human-readable citation string with [N] prefix and supporting evidence."""
    if isinstance(citation, OctusCitation):
        prefix = f"[{citation.ref_number}] " if citation.ref_number else ""
        display_text = citation.cited_text[:300] if citation.cited_text else ""
        excerpt = f' — "{display_text}"' if display_text else ""
        return (
            f"{prefix}[{citation.doc_source.upper()}] {citation.document_type} "
            f"({citation.document_date[:10]}){excerpt}"
        )
    elif isinstance(citation, SimFinCitation):
        prefix = f"[{citation.ref_number}] " if citation.ref_number else ""
        value = (
            f" = {citation.metric_value} ({citation.metric_unit})"
            if citation.metric_value
            else ""
        )
        return (
            f"{prefix}[SIMFIN] {citation.ticker} {citation.fiscal_period} "
            f"FY{citation.fiscal_year} — {citation.statement_type}: "
            f"{citation.metric_name}{value}"
        )
    return str(citation)


def format_citations_block(citations: list) -> str:
    """Format a list of citations as a markdown block, ordered by ref_number.

    SimFin citations are grouped by (ticker, statement_type) and rendered as
    markdown tables. Octus citations are listed individually with excerpts
    truncated to 300 chars for display.
    """
    if not citations:
        return "_No citations._"

    sorted_cites = sorted(citations, key=lambda c: c.ref_number if c.ref_number > 0 else 999)

    simfin = [c for c in sorted_cites if isinstance(c, SimFinCitation)]
    octus = [c for c in sorted_cites if isinstance(c, OctusCitation)]

    parts = []

    if simfin:
        groups: dict = defaultdict(list)
        for c in simfin:
            groups[(c.ticker, c.statement_type)].append(c)

        for (ticker, stmt), cits in groups.items():
            metrics = list(dict.fromkeys(c.metric_name for c in cits))
            header_cols = " | ".join(metrics)
            sep_cols = " | ".join(["---"] * len(metrics))
            parts.append(f"**{ticker} — {stmt.title()} Statement**")
            parts.append(f"| Ref | Period | {header_cols} |")
            parts.append(f"|-----|--------|{sep_cols}|")

            period_vals: dict = defaultdict(dict)
            period_refs: dict = defaultdict(list)
            for c in cits:
                pk = (c.fiscal_period, c.fiscal_year)
                period_vals[pk][c.metric_name] = c.metric_value
                if c.ref_number:
                    period_refs[pk].append(f"[{c.ref_number}]")

            for (period, year), vals in period_vals.items():
                refs = "".join(period_refs[(period, year)])
                row = " | ".join(vals.get(m, "—") for m in metrics)
                parts.append(f"| {refs} | {period} FY{year} | {row} |")

            parts.append("*Values: USD thousands (EPS: USD per share)*\n")

    if octus:
        if simfin:
            parts.append("---")
        for c in octus:
            parts.append(format_citation(c))

    return "\n".join(parts)
