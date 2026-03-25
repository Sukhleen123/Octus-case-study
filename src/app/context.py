"""AppContext: runtime objects returned by load_context()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AppContext:
    """
    Holds the live runtime objects for the application.

    Created by load_context() and cached via st.cache_resource in the Streamlit app.
    """

    retriever: Any   # MultiStoreRetriever (Pinecone-backed)
    graph: Any       # compiled LangGraph StateGraph
