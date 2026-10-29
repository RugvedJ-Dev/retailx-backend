from datetime import date

import pytest

from retailx_backend.agent.errors import PlanError
from retailx_backend.agent.mcp_gateway import McpGateway, ReadCache
from retailx_backend.agent.plan import Plan, PlanStep, requires_approval, validate_plan


def step(i, action, **kw):
    return PlanStep(id=f"s{i}", title=f"step {i}", action=action, **kw)


def agg(i, start="2026-01-01", end="2026-01-31", **kw):
    return step(i, "aggregate_sales", start_date=start, end_date=end, **kw)


def bad(*steps, match):
    with pytest.raises(PlanError, match=match):
        validate_plan(Plan(steps=list(steps)), 5)


def test_valid_compare_and_chart_plan_is_normalised():
    plan = Plan(steps=[agg(1), agg(2, "2025-12-01", "2025-12-31"),
                       step(3, "compare_periods", source_steps=["s1", "s2"]),
                       step(4, "build_chart", source_steps=["s3"])])
    steps = validate_plan(plan, 5)
    assert steps[0]["group_by"] == "none"
    assert steps[3]["chart_kind"] == "bar" and steps[3]["source_step"] == "s3"


def test_money_is_converted_to_paise_deterministically():
    (s,) = validate_plan(Plan(steps=[step(1, "insert_sale", date="2026-01-02", amount=100.5)]), 5)
    assert s["amount_paise"] == 10050


def test_rejects_invalid_plans():
    bad(*[agg(i) for i in range(1, 7)], match="at most 5")
    bad(step(2, "get_sales", date="2026-01-01"), match="s1..sN")
    bad(step(1, "get_sales", date="2026-02-30"), match="s1")
    bad(step(1, "get_sales", date="2026-01-01'; DROP"), match="s1")
    bad(agg(1), step(2, "compare_periods", source_steps=["s1", "s9"]), match="s2")
    bad(agg(1), step(2, "build_chart", source_steps=["s1"]), match="build_chart needs")  # ungrouped
    bad(step(1, "insert_sale", date="2026-01-01", amount=-5), match="positive")
    bad(step(1, "insert_sale", date="2026-01-01"), match="amount")
    bad(step(1, "update_sales", start_date="2026-01-01", end_date="2026-01-31",
             new_date="2026-02-01"), match="date range")
    bad(step(1, "update_sales", date="2026-01-01"), match="nothing to change")
    bad(step(1, "delete_sales"), match="date")
    bad(step(1, "sales_trend", start_date="2025-01-01", end_date="2026-01-01",
             granularity="day"), match="too many points")


def test_approval_policy_is_deterministic_code():
    ins, upd, dele, read = ({"action": a} for a in
                            ("insert_sale", "update_sales", "delete_sales", "aggregate_sales"))
    assert [requires_approval(s, "all_writes") for s in (ins, upd, dele, read)] == [True, True, True, False]
    assert [requires_approval(s, "destructive_only") for s in (ins, upd, dele, read)] == [
        False, True, True, False]


def test_read_cache_is_partitioned_by_user_and_invalidated():
    cache = ReadCache(60)
    cache.put("u1", "k", {"v": 1})
    assert cache.get("u1", "k") == {"v": 1} and cache.get("u2", "k") is None
    cache.invalidate("u1")
    assert cache.get("u1", "k") is None
    assert ReadCache(0).get("u", "k") is None


async def test_gateway_caches_reads_and_write_invalidates(settings, monkeypatch):
    calls = []

    async def fake_call(self, token, tool, args):
        calls.append(tool)
        return {"ok": True}

    monkeypatch.setattr(McpGateway, "_call", fake_call)
    gw = McpGateway(settings, "a" * 24, ReadCache(60))
    await gw.read("get_sales", {"date": "2026-01-01"})
    await gw.read("get_sales", {"date": "2026-01-01"})
    assert calls == ["get_sales"]
    await gw.write("insert_sale", {}, "op-12345678")
    await gw.read("get_sales", {"date": "2026-01-01"})
    assert calls == ["get_sales", "insert_sale", "get_sales"]


def test_today_param_type():
    assert isinstance(date.today(), date)
