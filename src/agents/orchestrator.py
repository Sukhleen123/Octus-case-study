"""
Orchestrator: coordinates Router, DocAgent, SimFinAgent, SynthesisAgent.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.doc_agent import DocAgent
from src.agents.router import route
from src.agents.simfin_agent import SimFinAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.citations.models import SynthesisResult

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Top-level coordinator for the multi-agent system.

    Roles:
      router → decides which agents to invoke
      doc_agent → retrieves Octus chunks
      simfin_agent → retrieves financial metrics
      synthesis_agent → merges results + produces final answer
    """

    def __init__(
        self,
        doc_agent: DocAgent,
        simfin_agent: SimFinAgent,
        synthesis_agent: SynthesisAgent,
        company_map: Any = None,  # DataFrame: octus_company_id -> ticker
    ) -> None:
        self._doc = doc_agent
        self._simfin = simfin_agent
        self._synthesis = synthesis_agent
        self._company_map = company_map

    def run(self, query: str, **retrieval_kwargs) -> SynthesisResult:
        """
        Process a user query end-to-end.

        Args:
            query: Natural language question.
            **retrieval_kwargs: Optional filters forwarded to DocAgent
              (company_name, doc_source, document_type, date_from, date_to).

        Returns:
            SynthesisResult with answer + citations + trace events.
        """
        decision = route(query)
        logger.info("Router decision for query '%s...': %s", query[:50], decision)

        # Extract all mentioned companies; resolve their SimFin tickers
        explicit_company = retrieval_kwargs.get("company_name")
        companies = (
            [explicit_company] if explicit_company
            else self._extract_companies_from_query(query)
        )
        tickers = self._resolve_tickers(companies)

        # Infer doc filters from query keywords; explicit retrieval_kwargs take precedence
        inferred = self._infer_doc_filters(query, companies)
        merged_kwargs = {**inferred, **retrieval_kwargs}

        # Invoke agents based on routing decision
        if decision == "octus":
            doc_result = self._doc.run(query, **merged_kwargs)
            simfin_result = ([], [], [])
        elif decision == "simfin":
            doc_result = ([], [], [])
            simfin_result = self._simfin.run(query, tickers)
        else:  # "both"
            doc_result = self._doc.run(query, **merged_kwargs)
            simfin_result = self._simfin.run(query, tickers)

        return self._synthesis.run(query, doc_result, simfin_result)

    def _extract_companies_from_query(self, query: str) -> list[str]:
        """Return ALL company names found in the query (substring + rapidfuzz fallback)."""
        if self._company_map is None:
            return []
        try:
            import pandas as pd
            from rapidfuzz import fuzz, process

            df = self._company_map
            if not isinstance(df, pd.DataFrame) or df.empty:
                return []

            known_names = df["company_name"].dropna().unique().tolist()
            q_lower = query.lower()
            matched: list[str] = []

            for name in known_names:
                words = [w for w in name.lower().split() if len(w) > 4]
                if name.lower() in q_lower or (words and any(w in q_lower for w in words)):
                    matched.append(name)

            # Fuzzy fallback only if nothing matched via substring
            if not matched:
                hits = process.extract(query, known_names, scorer=fuzz.partial_ratio, limit=3)
                matched = [name for name, score, _ in hits if score >= 70]

            for name in matched:
                logger.info("Extracted company from query: %s", name)
            return matched
        except Exception as e:
            logger.warning("Company extraction failed: %s", e)
        return []

    def _resolve_tickers(self, companies: list[str]) -> list[str]:
        """Look up SimFin tickers for all resolved companies."""
        if self._company_map is None or not companies:
            return []
        try:
            import pandas as pd
            df = self._company_map
            if not isinstance(df, pd.DataFrame) or df.empty:
                return []

            mask = df["company_name"].str.lower().isin([c.lower() for c in companies])
            subset = df[mask]
            tickers = (
                subset["suggested_ticker"]
                .dropna()
                .loc[lambda s: s.str.strip() != ""]
                .unique()
                .tolist()
            )
            return tickers[:10]  # cap to avoid excessive API calls
        except Exception as e:
            logger.warning("Could not resolve tickers: %s", e)
            return []

    _DOC_TYPE_KEYWORDS: dict[str, str] = {
        "10-k": "10-K",
        "annual report": "10-K",
        "10-q": "10-Q",
        "quarterly report": "10-Q",
    }
    _SEC_KEYWORDS = ("sec filing", "sec report", "form 10")

    def _infer_doc_filters(self, query: str, companies: list[str]) -> dict:
        """
        Infer Pinecone filters from query keywords.

        - document_type: "10-K" or "10-Q" when the query explicitly mentions it
        - doc_source: "sec_filing" when the query mentions SEC filings generically
        - company_name: set only when exactly one company is resolved
        """
        filters: dict = {}
        q_lower = query.lower()

        for kw, doc_type in self._DOC_TYPE_KEYWORDS.items():
            if kw in q_lower:
                filters["document_type"] = doc_type
                break

        if "document_type" not in filters:
            if any(kw in q_lower for kw in self._SEC_KEYWORDS):
                filters["doc_source"] = "sec_filing"

        if len(companies) == 1:
            filters["company_name"] = companies[0]

        return filters
