"""LangGraph nodes: planner -> (approval gate -> worker)* -> synthesize."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from ..domain import fmt_money
from .approvals import approval_id_for, build_payload
from .compute import compact_result, step_summary
from .deps import get_deps
from .errors import PlanError, StepError
from .executor import execute_step
from .plan import WRITE_ACTIONS, requires_approval, validate_plan
from .state import AgentState

log = logging.getLogger("retailx.agent")

LLM_DOWN = "I couldn't reach the AI model just now. Please try again in a moment."
BAD_PLAN = ("I couldn't turn that into a safe set of operations. Could you rephrase it, for "
            "example with explicit dates?")
UNCLEAR = "I'm not sure what you'd like me to do. Could you rephrase that?"


def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else " ".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


# ------------------------------------------------------------------------------ planner
async def planner_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps, write = get_deps(config), get_stream_writer()
    messages = state["messages"]
    request = _text(messages[-1])
    history = [("user" if m.type == "human" else "assistant", _text(m))
               for m in messages[:-1][-deps.settings.history_window:]]
    today = datetime.now(ZoneInfo(deps.settings.timezone)).date()
    errors: list[str] | None = None
    steps: list[dict[str, Any]] | None = None
    plan = None
    for _attempt in range(2):
        try:
            plan = await deps.llm.plan(
                today=today, tz=deps.settings.timezone, max_steps=deps.settings.max_plan_steps,
                history=history, request=request, errors=errors)
        except Exception:
            log.exception("planner LLM call failed")
            return {"plan": [], "direct_reply": LLM_DOWN, "error": "llm_unavailable"}
        try:
            steps = validate_plan(plan, deps.settings.max_plan_steps)
            break
        except PlanError as exc:
            errors = exc.errors
            log.warning("invalid plan, retrying: %s", exc.errors)
    if steps is None or plan is None:
        return {"plan": [], "direct_reply": BAD_PLAN, "error": "invalid_plan"}
    if not steps:
        return {"plan": [], "direct_reply": (plan.direct_reply or "").strip() or UNCLEAR}
    policy = deps.settings.approval_policy
    write({"type": "plan", "steps": [
        {"id": s["id"], "title": s["title"], "action": s["action"],
         "requires_approval": requires_approval(s, policy)} for s in steps]})
    return {"plan": steps, "direct_reply": None}


def route(state: AgentState, config: RunnableConfig) -> str:
    """Explicit, bounded state transitions after the planner, gate and worker nodes."""
    if state.get("aborted"):
        return "synthesize"
    plan, cursor = state.get("plan") or [], state.get("cursor", 0)
    if cursor >= len(plan):
        return "synthesize"
    step = plan[cursor]
    policy = get_deps(config).settings.approval_policy
    if requires_approval(step, policy) and step["id"] not in (state.get("approvals") or {}):
        return "approval_gate"
    return "worker"


# ------------------------------------------------------------------------- approval gate
async def approval_gate_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Pause for human approval.

    LangGraph re-executes this node from the top on resume, so everything before ``interrupt``
    is read-only and the approval id is deterministic.
    """
    deps, write = get_deps(config), get_stream_writer()
    cursor = state["cursor"]
    step = state["plan"][cursor]
    sid = step["id"]
    approvals, results = dict(state.get("approvals") or {}), dict(state.get("results") or {})
    preview = None
    if step["action"] in ("update_sales", "delete_sales"):
        try:
            preview = await deps.mcp.read("preview_sales_selection", {"selector": step["selector"]})
        except StepError as exc:
            results[sid] = {"status": "failed", "action": step["action"], "error": str(exc)}
            return {"results": results, "aborted": True}
        if preview["count"] == 0:  # nothing to change: don't bother the user
            approvals[sid] = "skipped"
            results[sid] = {"status": "ok", "action": step["action"],
                            "data": {"affected": 0, "total_paise": 0, "total": 0.0, "skipped": True}}
            return {"approvals": approvals, "results": results, "cursor": cursor + 1}
    approval_id = approval_id_for(state["run_id"], sid)
    answer = interrupt(build_payload(step, preview, approval_id))
    approved = (isinstance(answer, dict) and answer.get("decision") == "approve"
                and answer.get("approval_id") == approval_id)
    if approved:
        approvals[sid] = "approved"
        write({"type": "approval_resolved", "decision": "approve", "step_id": sid})
        return {"approvals": approvals}
    approvals[sid] = "rejected"
    results[sid] = {"status": "rejected", "action": step["action"]}
    write({"type": "approval_resolved", "decision": "reject", "step_id": sid})
    return {"approvals": approvals, "results": results, "aborted": True}


# ----------------------------------------------------------------------------- worker
async def worker_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps, write = get_deps(config), get_stream_writer()
    cursor = state["cursor"]
    step = state["plan"][cursor]
    sid, action = step["id"], step["action"]
    results, charts = dict(state.get("results") or {}), list(state.get("charts") or [])
    write({"type": "step_started", "id": sid, "title": step["title"], "action": action})
    started = time.perf_counter()
    try:
        needs_ok = requires_approval(step, deps.settings.approval_policy)
        if needs_ok and (state.get("approvals") or {}).get(sid) != "approved":
            # Fail closed: a write the policy gates never runs without a recorded approval.
            raise StepError("That change was not approved, so it was not applied.")
        op_key = f"{deps.thread_id}:{state['run_id']}:{sid}"
        data = await execute_step(step, results, deps, op_key)
    except StepError as exc:
        results[sid] = {"status": "failed", "action": action, "error": str(exc)}
        write({"type": "step_failed", "id": sid, "title": step["title"], "error": str(exc)})
        return {"results": results, "aborted": True}
    except Exception:
        log.exception("unexpected error in step %s (%s)", sid, action)
        msg = "Something went wrong while running this step."
        results[sid] = {"status": "failed", "action": action, "error": msg}
        write({"type": "step_failed", "id": sid, "title": step["title"], "error": msg})
        return {"results": results, "aborted": True}
    results[sid] = {"status": "ok", "action": action, "data": data}
    if action == "build_chart" and data.get("type"):
        charts.append(data)
        write({"type": "chart", "chart": data})
    log.info("step done id=%s action=%s ms=%d", sid, action, (time.perf_counter() - started) * 1000)
    write({"type": "step_completed", "id": sid, "title": step["title"],
           "summary": step_summary(step, data)})
    return {"results": results, "charts": charts, "cursor": cursor + 1}


# ------------------------------------------------------------------------- synthesize
def choose_tier(plan: list[dict[str, Any]]) -> str:
    """Use the capable model only for multi-step / analytical answers."""
    complex_actions = {"compare_periods", "sales_trend"}
    return "strong" if len(plan) >= 3 or any(s["action"] in complex_actions for s in plan) else "fast"


def template_reply(state: AgentState) -> str | None:
    """Deterministic replies (no LLM call) for rejections, failures and plain write confirmations."""
    plan, results = state.get("plan") or [], state.get("results") or {}
    if any(r.get("status") == "rejected" for r in results.values()):
        return "Understood - I cancelled that change. Nothing was modified."
    failed = [r for r in results.values() if r.get("status") == "failed"]
    if failed:
        return f"Sorry, I couldn't finish that: {failed[0]['error']}"
    if plan and all(s["action"] in WRITE_ACTIONS for s in plan):
        lines = []
        for s in plan:
            data = results[s["id"]]["data"]
            if data.get("skipped") or data.get("affected") == 0:
                lines.append("No matching sales records were found, so nothing was changed.")
            elif s["action"] == "insert_sale":
                lines.append(f"Done - added {fmt_money(s['amount_paise'])} to sales on {s['date']}.")
            elif s["action"] == "update_sales":
                lines.append(f"Done - updated {data['affected']} sales record(s).")
            else:
                lines.append(f"Done - deleted {data['affected']} sales record(s) totalling "
                             f"{fmt_money(data['total_paise'])}.")
        return " ".join(lines)
    return None


def build_facts(state: AgentState) -> dict[str, Any]:
    results = state.get("results") or {}
    return {"steps": [
        {"title": s["title"], "result": compact_result(s, results[s["id"]]["data"])}
        for s in state.get("plan") or [] if results.get(s["id"], {}).get("status") == "ok"
    ], "charts_built": [c["title"] for c in state.get("charts") or []]}


async def synthesize_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps, write = get_deps(config), get_stream_writer()
    text = state.get("direct_reply") or template_reply(state)
    if text is None:
        request = _text([m for m in state["messages"] if m.type == "human"][-1])
        facts = build_facts(state)
        parts: list[str] = []
        try:
            async for piece in deps.llm.synthesize(
                    tier=choose_tier(state["plan"]), request=request, facts=facts):
                parts.append(piece)
                write({"type": "token", "text": piece})
            text = "".join(parts).strip()
        except Exception:
            log.exception("synthesis LLM call failed")
        if not text:  # graceful degradation: deterministic summary of what was computed
            lines = [f"{s['title']}: {step_summary(s, state['results'][s['id']]['data'])}"
                     for s in state["plan"] if state["results"].get(s["id"], {}).get("status") == "ok"]
            text = "Here's what I found. " + "; ".join(lines) + "."
            write({"type": "token", "text": text})
    else:
        write({"type": "token", "text": text})
    return {"final_text": text, "messages": [AIMessage(content=text)]}
