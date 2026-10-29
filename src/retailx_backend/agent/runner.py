"""Drives the graph for one turn and turns LangGraph output into client-facing events.

Only concise execution events and user-facing content are emitted; model reasoning is never
forwarded.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from bson import ObjectId
from langgraph.types import Command

from .. import repos
from ..config import Settings
from .deps import Deps
from .llm import LLMGateway
from .mcp_gateway import McpGateway, ReadCache
from .state import new_run_input

log = logging.getLogger("retailx.runner")
GENERIC_ERROR = "Something went wrong while processing your request. Please try again."


def approval_view(doc: dict[str, Any]) -> dict[str, Any]:
    p = doc["payload"]
    return {"approval_id": doc["_id"], "thread_id": str(doc["thread_id"]), "status": doc["status"],
            "operation": p["operation"], "summary": p["summary"], "details": p["details"],
            "destructive": p["destructive"], "warnings": p["warnings"],
            "expires_at": doc["expires_at"].isoformat()}


def message_view(doc: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(doc["_id"]), "role": doc["role"], "content": doc["content"],
            "charts": doc.get("charts", []), "created_at": doc["created_at"].isoformat()}


class AgentRunner:
    def __init__(self, *, settings: Settings, db: Any, graph: Any, llm: LLMGateway,
                 cache: ReadCache | None = None) -> None:
        self.settings, self.db, self.graph, self.llm = settings, db, graph, llm
        self.cache = cache or ReadCache(settings.read_cache_ttl_seconds)

    def _config(self, user_id: ObjectId, thread_oid: ObjectId) -> dict[str, Any]:
        deps = Deps(self.settings, self.llm, McpGateway(self.settings, str(user_id), self.cache),
                    str(user_id), str(thread_oid))
        # recursion_limit bounds total node executions (plan <= 5 steps => ~12 nodes).
        return {"configurable": {"thread_id": str(thread_oid), "deps": deps}, "recursion_limit": 40}

    async def pending_interrupt(self, user_id: ObjectId, thread_oid: ObjectId) -> dict[str, Any] | None:
        """The approval payload the graph is currently paused on, if any (source of truth)."""
        snapshot = await self.graph.aget_state(self._config(user_id, thread_oid))
        return snapshot.interrupts[0].value if snapshot.interrupts else None

    async def delete_state(self, thread_id: str) -> None:
        """Remove a thread's LangGraph checkpoints."""
        try:
            await asyncio.to_thread(self.graph.checkpointer.delete_thread, thread_id)
        except Exception:
            log.exception("failed to delete checkpoints thread=%s", thread_id)

    async def run_chat(self, user_id: ObjectId, thread_oid: ObjectId, message: str) -> AsyncIterator[dict]:
        run_id = uuid.uuid4().hex
        yield {"type": "run_started", "run_id": run_id}
        async for event in self._drive(new_run_input(message, run_id), user_id, thread_oid):
            yield event

    async def run_resume(
        self, user_id: ObjectId, thread_oid: ObjectId, approval_id: str, decision: str
    ) -> AsyncIterator[dict]:
        yield {"type": "run_started", "run_id": approval_id}
        command = Command(resume={"decision": decision, "approval_id": approval_id})
        ok = True
        async for event in self._drive(command, user_id, thread_oid):
            ok = ok and event["type"] != "error"
            yield event
        if ok:
            await repos.finish_approval(self.db, approval_id,
                                        "approved" if decision == "approve" else "rejected")

    async def _drive(self, graph_input: Any, user_id: ObjectId, thread_oid: ObjectId) -> AsyncIterator[dict]:
        config = self._config(user_id, thread_oid)
        try:
            async for chunk in self.graph.astream(graph_input, config, stream_mode="custom"):
                if isinstance(chunk, dict):
                    yield chunk
            snapshot = await self.graph.aget_state(config)
            if snapshot.interrupts:
                payload = snapshot.interrupts[0].value
                doc = await repos.upsert_approval(
                    self.db, approval_id=payload["approval_id"], thread_oid=thread_oid,
                    user_id=user_id, run_id=snapshot.values.get("run_id", ""), payload=payload,
                    ttl_hours=self.settings.approval_ttl_hours)
                await repos.add_message(self.db, thread_oid=thread_oid, user_id=user_id,
                                        role="assistant",
                                        content=f"I need your approval to: {payload['summary']}")
                await repos.touch_thread(self.db, thread_oid)
                yield {"type": "approval_required", "approval": approval_view(doc)}
            else:
                values = snapshot.values
                doc = await repos.add_message(
                    self.db, thread_oid=thread_oid, user_id=user_id, role="assistant",
                    content=values.get("final_text") or "", charts=values.get("charts") or [])
                await repos.touch_thread(self.db, thread_oid)
                yield {"type": "message_complete", "message": message_view(doc)}
        except Exception:
            log.exception("agent run failed thread=%s", thread_oid)
            yield {"type": "error", "code": "agent_error", "message": GENERIC_ERROR}
