"""Graph assembly."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import approval_gate_node, planner_node, route, synthesize_node, worker_node
from .state import AgentState

_TARGETS = ["approval_gate", "worker", "synthesize"]


def build_graph(checkpointer: Any):
    """Compile the Planner-Worker graph with a persistent checkpointer (MongoDB in production)."""
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("approval_gate", approval_gate_node)
    g.add_node("worker", worker_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "planner")
    for node in ("planner", "approval_gate", "worker"):
        g.add_conditional_edges(node, route, _TARGETS)
    g.add_edge("synthesize", END)
    return g.compile(checkpointer=checkpointer)
