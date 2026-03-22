"""AppContext: runtime objects returned by bootstrap.ensure_ready()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


@dataclass
class AppContext:
    """
    Holds all live runtime objects for the application.

    Created by bootstrap.ensure_ready(settings) and cached via
    st.cache_resource in the Streamlit app.
    """

    vectorstore: Any          # PineconeStore | FAISSStore
    retriever: Any            # DenseRetriever | DenseMMRRetriever
    simfin_client: Any        # SimFinV3Client | None (None in batch-only mode)
    orchestrator: Any         # Orchestrator
    settings_fingerprint: str # fingerprint at build time — used to detect staleness
