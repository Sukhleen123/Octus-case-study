"""
Runtime singletons for the agent graph.

Call runtime.init() once at app startup (from src/app/loader.py) before
the graph is invoked. Node functions import these directly.
"""

from __future__ import annotations

from src.app.settings import Settings

settings: Settings = Settings()
retriever = None
llm_client = None
company_map = None


def init(retriever_, llm_client_, company_map_=None) -> None:
    """Initialize runtime singletons. Called once by loader.py at startup."""
    global retriever, llm_client, company_map
    retriever = retriever_
    llm_client = llm_client_
    company_map = company_map_
