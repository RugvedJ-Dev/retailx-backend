"""Per-run dependencies, passed through LangGraph's ``configurable`` (never checkpointed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings
from .llm import LLMGateway
from .mcp_gateway import McpGateway


@dataclass
class Deps:
    settings: Settings
    llm: LLMGateway
    mcp: McpGateway
    user_id: str  # established by the API from the verified session, never by the model
    thread_id: str


def get_deps(config: dict[str, Any]) -> Deps:
    return config["configurable"]["deps"]
