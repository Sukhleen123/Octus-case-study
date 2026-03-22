"""
Chainlit UI — Financial Multi-Agent Intelligence System

Default workflow:
    chainlit run apps/chainlit_app.py

Shows: agent steps, retrieved sources, structured citations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chainlit as cl

from src.app.bootstrap import ensure_ready
from src.app.logging import configure_logging
from src.app.settings import Settings
from src.citations.formatter import format_citation

configure_logging()


# ── App startup: runs once when Chainlit server starts ────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Initialize AppContext and store in user session."""
    settings = Settings()

    await cl.Message(content="Initializing system (first run may take a moment)...").send()

    try:
        ctx = ensure_ready(settings)
        cl.user_session.set("ctx", ctx)
        cl.user_session.set("settings", settings)
        await cl.Message(
            content=(
                f"Ready! Vectorstore: `{type(ctx.vectorstore).__name__}` | "
                f"Embedder: `{settings.embedding_provider}` | "
                f"SimFin mode: `{settings.simfin_mode}`"
            )
        ).send()
    except Exception as e:
        await cl.Message(content=f"Bootstrap error: {e}").send()


# ── Message handler ────────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    """Process a user query through the orchestrator."""
    ctx = cl.user_session.get("ctx")
    if ctx is None:
        await cl.Message(content="System not initialized. Please restart.").send()
        return

    query = message.content

    # Show agent steps as they are emitted
    async with cl.Step(name="Orchestrator", type="run") as step:
        step.input = query

        try:
            result = ctx.orchestrator.run(query)
        except Exception as e:
            step.output = f"Error: {e}"
            await cl.Message(content=f"Error: {e}").send()
            return

        step.output = f"Routing: complete | Chunks: {len(result.citations)} citations"

    # Render trace events as nested steps
    for event in result.trace_events:
        event_type = event.get("event_type", "")
        agent = event.get("agent_name", "")

        if event_type == "retrieval_results":
            count = event.get("payload", {}).get("count", 0)
            async with cl.Step(name=f"{agent}: retrieved {count} chunks", type="retrieval"):
                pass

    # Final answer
    await cl.Message(content=result.final_answer_text).send()

    # Citations as a separate element
    if result.citations:
        citation_lines = []
        for i, c in enumerate(result.citations, 1):
            citation_lines.append(f"{i}. {format_citation(c)}")

        citations_text = "\n".join(citation_lines)
        await cl.Message(
            content=f"**Sources:**\n{citations_text}",
            author="citations",
        ).send()
