"""LangGraph state schema and per-run input."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # persisted conversation memory
    run_id: str
    plan: list[dict[str, Any]]  # validated steps
    direct_reply: str | None
    cursor: int  # index of the next step to execute
    results: dict[str, dict[str, Any]]  # step id -> {status, action, data|error}
    approvals: dict[str, str]  # step id -> approved | rejected | skipped
    charts: list[dict[str, Any]]
    aborted: bool
    final_text: str
    error: str | None


def new_run_input(message: str, run_id: str) -> dict[str, Any]:
    """Per-turn input: appends the user message and resets all per-run fields."""
    return {
        "messages": [HumanMessage(content=message)], "run_id": run_id, "plan": [],
        "direct_reply": None, "cursor": 0, "results": {}, "approvals": {}, "charts": [],
        "aborted": False, "final_text": "", "error": None,
    }
