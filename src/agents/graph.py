"""
Agent graph: wires all nodes into a compiled LangGraph StateGraph.

Call build_graph() once at startup. The returned compiled graph is
invoked per query via graph.invoke() or graph.stream().

Runtime dependencies (retriever, llm_client, etc.) are accessed
directly from src.agents.runtime — initialize them first via runtime.init().
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.doc_agent import doc_agent_node
from src.agents.router import router_node
from src.agents.simfin_agent import simfin_agent_node
from src.agents.state import AgentState
from src.agents.synthesis_agent import synthesize_node


def route_to_agents(state: AgentState) -> list[str]:
    """
    Conditional edge after the router node.

    Returns a list so LangGraph fans out concurrently when both agents are needed.
    """
    if state.route == "both":
        return ["doc_agent", "simfin_agent"]
    if state.route == "octus":
        return ["doc_agent"]
    return ["simfin_agent"]


def build_graph():
    """Assemble and compile the agent StateGraph. Call once at startup."""
    graph = StateGraph(AgentState)

    graph.add_node("router",       router_node)
    graph.add_node("doc_agent",    doc_agent_node)
    graph.add_node("simfin_agent", simfin_agent_node)
    graph.add_node("synthesize",   synthesize_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_to_agents)

    # Both retrieval branches converge on the synthesis node (fan-in)
    graph.add_edge("doc_agent",    "synthesize")
    graph.add_edge("simfin_agent", "synthesize")
    graph.add_edge("synthesize",   END)

    return graph.compile()
