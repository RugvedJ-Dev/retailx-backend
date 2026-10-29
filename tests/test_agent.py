from bson import ObjectId
from langgraph.checkpoint.memory import InMemorySaver

import pytest

from retailx_backend import repos
from retailx_backend.agent.graph import build_graph
from retailx_backend.agent.nodes import BAD_PLAN, LLM_DOWN
from retailx_backend.agent.plan import Plan, PlanStep
from retailx_backend.agent.runner import AgentRunner


class ScriptedLLM:
    """Deterministic stand-in for the LLM gateway."""

    def __init__(self, plans, text="Here is your answer."):
        self.plans, self.text = list(plans), text
        self.plan_calls, self.synth_calls = [], []

    async def plan(self, **kwargs):
        self.plan_calls.append(kwargs)
        item = self.plans.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def synthesize(self, *, tier, request, facts):
        self.synth_calls.append((tier, facts))
        for word in self.text.split(" "):
            yield word + " "


def S(i, action, **kw):
    return PlanStep(id=f"s{i}", title=f"step {i}", action=action, **kw)


def P(*steps, reply=None):
    return Plan(steps=list(steps), direct_reply=reply)


async def collect(agen):
    return [event async for event in agen]


def types(events):
    return [e["type"] for e in events]


class Env:
    def __init__(self, settings, db, mcp_url, saver, store, user):
        self.settings = settings.model_copy(update={"mcp_url": mcp_url, "mcp_timeout_seconds": 5})
        self.db, self.saver, self.store, self.user = db, saver, store, user
        self.uid = ObjectId(user)
        self.tid = None

    async def init(self):
        self.tid = (await repos.create_thread(self.db, self.uid, "t"))["_id"]
        return self

    def runner(self, llm, **updates):
        settings = self.settings.model_copy(update=updates) if updates else self.settings
        return AgentRunner(settings=settings, db=self.db, graph=build_graph(self.saver), llm=llm)

    async def chat(self, runner, text="hi"):
        return await collect(runner.run_chat(self.uid, self.tid, text))

    async def decide(self, runner, decision, approval_id=None):
        payload = await runner.pending_interrupt(self.uid, self.tid)
        approval_id = approval_id or payload["approval_id"]
        await repos.claim_approval(self.db, approval_id=payload["approval_id"], thread_oid=self.tid,
                                   user_id=self.uid, decision=decision)
        return await collect(runner.run_resume(self.uid, self.tid, approval_id, decision))

    async def sales(self, day="2026-10-29", user=None):
        return await self.store.sales_for_date(user or self.user, day)


@pytest.fixture
async def env(settings, db, mcp_url, store, user_a):
    return await Env(settings, db, mcp_url, InMemorySaver(), store, user_a).init()


async def seed(store, user, rows):
    for i, (day, paise) in enumerate(rows):
        await store.insert_sale(user, day, paise, "", f"seed-{user[-6:]}-{i:04d}")


# ---------------------------------------------------------------------------- reads
async def test_read_flow_streams_events_and_persists_history(env):
    await seed(env.store, env.user, [("2026-10-02", 10000), ("2026-10-05", 5000)])
    llm = ScriptedLLM([P(S(1, "aggregate_sales", start_date="2026-10-01", end_date="2026-10-29"))])
    events = await env.chat(env.runner(llm), "What were my sales this month?")
    assert types(events) == ["run_started", "plan", "step_started", "step_completed", "token",
                             "token", "token", "token", "message_complete"]
    assert events[3]["summary"] == "\u20b9150.00 across 2 sale(s)"
    assert llm.synth_calls[0][0] == "fast"  # single simple step -> lightweight model
    assert llm.synth_calls[0][1]["steps"][0]["result"]["total"] == "\u20b9150.00"
    msgs = await repos.list_messages(env.db, env.tid, env.uid)
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == events[-1]["message"]["content"]


async def test_compare_and_chart_math_is_deterministic_and_uses_strong_tier(env):
    await seed(env.store, env.user, [("2026-10-02", 300000), ("2026-09-10", 200000)])
    llm = ScriptedLLM([P(
        S(1, "aggregate_sales", start_date="2026-10-01", end_date="2026-10-31"),
        S(2, "aggregate_sales", start_date="2026-09-01", end_date="2026-09-30"),
        S(3, "compare_periods", source_steps=["s1", "s2"]),
        S(4, "build_chart", source_steps=["s3"], chart_title="This vs last month"))])
    events = await env.chat(env.runner(llm), "compare with last month and chart it")
    tier, facts = llm.synth_calls[0]
    assert tier == "strong"
    cmp_ = facts["steps"][2]["result"]
    assert cmp_["pct_change"] == 50.0 and cmp_["difference"] == "\u20b91,000.00"
    assert (cmp_["current"]["label"], cmp_["baseline"]["label"]) == ("October 2026", "September 2026")
    chart = next(e for e in events if e["type"] == "chart")["chart"]
    assert chart["type"] == "bar" and [d["total"] for d in chart["data"]] == [2000.0, 3000.0]
    assert events[-1]["message"]["charts"][0]["id"] == chart["id"]


async def test_trend_chart_for_visualisation(env):
    await seed(env.store, env.user, [("2026-01-02", 1000), ("2026-01-04", 3000)])
    llm = ScriptedLLM([P(S(1, "sales_trend", start_date="2026-01-01", end_date="2026-01-05"),
                         S(2, "build_chart", source_steps=["s1"], chart_title="January"))])
    events = await env.chat(env.runner(llm), "visualize january")
    chart = next(e for e in events if e["type"] == "chart")["chart"]
    assert chart["type"] == "line" and [d["total"] for d in chart["data"]] == [0, 10.0, 0, 30.0, 0]


async def test_direct_reply_needs_no_tools_or_synthesis(env):
    llm = ScriptedLLM([P(reply="Hello! Ask me about your sales.")])
    events = await env.chat(env.runner(llm))
    assert events[-1]["message"]["content"] == "Hello! Ask me about your sales."
    assert llm.synth_calls == [] and "step_started" not in types(events)


async def test_memory_is_passed_to_planner_on_later_turns(env):
    llm = ScriptedLLM([P(reply="First answer"), P(reply="Second answer")])
    runner = env.runner(llm)
    await env.chat(runner, "first question")
    await env.chat(runner, "second question")
    assert llm.plan_calls[1]["history"] == [("user", "first question"), ("assistant", "First answer")]


# --------------------------------------------------------------------- plan failures
async def test_invalid_plan_is_retried_with_errors_then_succeeds(env):
    broken = P(S(1, "get_sales", date="2026-02-30"))
    fixed = P(reply="ok")
    llm = ScriptedLLM([broken, fixed])
    events = await env.chat(env.runner(llm))
    assert llm.plan_calls[0]["errors"] is None and "s1" in llm.plan_calls[1]["errors"][0]
    assert events[-1]["message"]["content"] == "ok"


async def test_two_invalid_plans_fall_back_safely(env):
    broken = P(S(1, "get_sales", date="nope"))
    events = await env.chat(env.runner(ScriptedLLM([broken, broken])))
    assert events[-1]["message"]["content"] == BAD_PLAN and "step_started" not in types(events)


async def test_llm_outage_degrades_gracefully(env):
    events = await env.chat(env.runner(ScriptedLLM([RuntimeError("boom")])))
    assert events[-1]["message"]["content"] == LLM_DOWN and "error" not in types(events)


async def test_synthesis_failure_falls_back_to_deterministic_summary(env):
    await seed(env.store, env.user, [("2026-10-02", 10000)])
    llm = ScriptedLLM([P(S(1, "aggregate_sales", start_date="2026-10-01", end_date="2026-10-29"))])

    async def broken(**kwargs):
        raise RuntimeError("llm down")
        yield  # pragma: no cover

    llm.synthesize = broken
    events = await env.chat(env.runner(llm))
    assert "\u20b9100.00" in events[-1]["message"]["content"]


async def test_tool_outage_is_reported_and_stops_the_plan(env):
    llm = ScriptedLLM([P(S(1, "aggregate_sales", start_date="2026-10-01", end_date="2026-10-29"),
                         S(2, "aggregate_sales", start_date="2026-09-01", end_date="2026-09-30"))])
    env.settings = env.settings.model_copy(update={"mcp_url": "http://127.0.0.1:9/mcp",
                                                   "mcp_timeout_seconds": 1})
    events = await env.chat(env.runner(llm))
    assert "step_failed" in types(events) and types(events).count("step_started") == 1
    assert "unavailable" in events[-1]["message"]["content"]


async def test_tool_validation_errors_surface_without_crashing(env):
    # amount valid for the planner but the MCP selector rejects an oversized bulk op
    await seed(env.store, env.user, [("2026-10-01", 100)] * 6)  # store.max_bulk == 5
    llm = ScriptedLLM([P(S(1, "delete_sales", start_date="2026-10-01", end_date="2026-10-31"))])
    runner = env.runner(llm)
    first = await env.chat(runner)
    assert types(first)[-1] == "approval_required"
    events = await env.decide(runner, "approve")
    assert "max 5" in events[-1]["message"]["content"]
    assert (await env.sales("2026-10-01"))["total_count"] == 6


# -------------------------------------------------------------------------- HITL
async def test_insert_pauses_for_approval_then_executes_exactly_once(env):
    llm = ScriptedLLM([P(S(1, "insert_sale", date="2026-10-29", amount=100, note="cash"))])
    runner = env.runner(llm)
    events = await env.chat(runner, "Add Rs 100 to today's sales")
    approval = events[-1]["approval"]
    assert events[-1]["type"] == "approval_required" and "message_complete" not in types(events)
    assert approval["summary"] == "Add a sale of \u20b9100.00 on 2026-10-29" and not approval["destructive"]
    assert (await env.sales())["total_count"] == 0  # nothing written before approval
    doc = await repos.get_pending_approval(env.db, env.tid, env.uid)
    assert doc["_id"] == approval["approval_id"]

    resumed = await env.decide(runner, "approve")
    assert "approval_resolved" in types(resumed) and types(resumed)[-1] == "message_complete"
    assert resumed[-1]["message"]["content"] == "Done - added \u20b9100.00 to sales on 2026-10-29."
    assert (await env.sales())["total_count"] == 1
    assert (await repos.get_approval(env.db, approval["approval_id"], env.uid))["status"] == "approved"

    # duplicate/retried approval: no claim possible, nothing pending in the graph
    again = await repos.claim_approval(env.db, approval_id=approval["approval_id"],
                                       thread_oid=env.tid, user_id=env.uid, decision="approve")
    assert again is None and await runner.pending_interrupt(env.uid, env.tid) is None
    assert (await env.sales())["total_count"] == 1


async def test_rejection_never_executes(env):
    llm = ScriptedLLM([P(S(1, "insert_sale", date="2026-10-29", amount=100))])
    runner = env.runner(llm)
    await env.chat(runner)
    resumed = await env.decide(runner, "reject")
    assert "cancelled" in resumed[-1]["message"]["content"]
    assert (await env.sales())["total_count"] == 0


async def test_approval_for_a_different_id_is_treated_as_rejection(env):
    llm = ScriptedLLM([P(S(1, "insert_sale", date="2026-10-29", amount=100))])
    runner = env.runner(llm)
    await env.chat(runner)
    resumed = await env.decide(runner, "approve", approval_id="forged-approval-id")
    assert "cancelled" in resumed[-1]["message"]["content"]
    assert (await env.sales())["total_count"] == 0


async def test_pending_approval_survives_a_restart(env):
    llm = ScriptedLLM([P(S(1, "insert_sale", date="2026-10-29", amount=250))])
    await env.chat(env.runner(llm))
    restarted = env.runner(ScriptedLLM([]))  # new graph/runner/gateway, same persisted checkpoints
    resumed = await env.decide(restarted, "approve")
    assert types(resumed)[-1] == "message_complete"
    assert (await env.sales())["total_paise"] == 25000


async def test_delete_shows_preview_and_only_touches_own_data(env, user_b):
    await seed(env.store, env.user, [("2026-10-01", 1000), ("2026-10-02", 2000), ("2026-10-03", 3000)])
    await seed(env.store, user_b, [("2026-10-01", 9999)])
    llm = ScriptedLLM([P(S(1, "delete_sales", start_date="2026-10-01", end_date="2026-10-31"))])
    runner = env.runner(llm)
    approval = (await env.chat(runner))[-1]["approval"]
    assert approval["destructive"] and approval["details"]["records"] == 3
    assert approval["details"]["total"] == "\u20b960.00" and "3 sales record(s)" in approval["summary"]
    resumed = await env.decide(runner, "approve")
    assert "deleted 3" in resumed[-1]["message"]["content"]
    assert (await env.sales("2026-10-02"))["total_count"] == 0
    assert (await env.sales("2026-10-01", user_b))["total_paise"] == 9999


async def test_update_with_nothing_to_change_skips_approval(env):
    llm = ScriptedLLM([P(S(1, "update_sales", date="2026-10-01", new_amount=5))])
    events = await env.chat(env.runner(llm))
    assert "approval_required" not in types(events)
    assert "nothing was changed" in events[-1]["message"]["content"]


async def test_destructive_only_policy_lets_single_inserts_through(env):
    llm = ScriptedLLM([P(S(1, "insert_sale", date="2026-10-29", amount=100))])
    events = await env.chat(env.runner(llm, approval_policy="destructive_only"))
    assert "approval_required" not in types(events) and "Done - added" in events[-1]["message"]["content"]
    assert (await env.sales())["total_count"] == 1


async def test_two_writes_require_two_separate_approvals(env):
    llm = ScriptedLLM([P(S(1, "insert_sale", date="2026-10-29", amount=100),
                         S(2, "insert_sale", date="2026-10-29", amount=200))])
    runner = env.runner(llm)
    first = (await env.chat(runner))[-1]["approval"]
    second_events = await env.decide(runner, "approve")
    second = second_events[-1]["approval"]
    assert first["approval_id"] != second["approval_id"] and (await env.sales())["total_paise"] == 10000
    final = await env.decide(runner, "approve")
    assert types(final)[-1] == "message_complete" and (await env.sales())["total_paise"] == 30000
