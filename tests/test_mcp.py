import jwt
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from retailx_backend.security import mint_mcp_token

R = ["sales:read"]
W = ["sales:write"]


def rd(settings, user):
    return mint_mcp_token(settings, user, R)


def wr(settings, user, op):
    return mint_mcp_token(settings, user, W, op_key=op)


async def seed(store, user, rows):
    for i, (day, paise) in enumerate(rows):
        await store.insert_sale(user, day, paise, "", f"seed-{user[-6:]}-{i:04d}")


async def call(url, token, tool, args):
    async with Client(url, auth=token) as client:
        return (await client.call_tool(tool, args)).structured_content


async def test_tools_declare_schemas_and_never_take_user_id(mcp_url, settings, user_a):
    async with Client(mcp_url, auth=rd(settings, user_a)) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert set(tools) == {"get_sales", "get_sales_range", "aggregate_sales", "get_sales_trend",
                          "preview_sales_selection", "insert_sale", "update_sales",
                          "delete_sales"}
    for tool in tools.values():
        assert tool.description and tool.outputSchema
        assert "user_id" not in str(tool.inputSchema)


async def test_unauthenticated_and_forged_tokens_are_rejected(mcp_url, user_a):
    with pytest.raises(Exception):
        await call(mcp_url, None, "get_sales", {"date": "2026-01-01"})
    forged = jwt.encode({"sub": user_a, "scope": "sales:read", "aud": "retailx-mcp",
                         "iss": "retailx-api", "exp": 9999999999}, "x" * 40, algorithm="HS256")
    with pytest.raises(Exception):
        await call(mcp_url, forged, "get_sales", {"date": "2026-01-01"})


async def test_reads_are_isolated_per_user(mcp_url, settings, store, user_a, user_b):
    await seed(store, user_a, [("2026-01-05", 10000), ("2026-01-06", 5000)])
    await seed(store, user_b, [("2026-01-05", 99900)])
    a = await call(mcp_url, rd(settings, user_a), "aggregate_sales",
                   {"start_date": "2026-01-01", "end_date": "2026-01-31"})
    b = await call(mcp_url, rd(settings, user_b), "aggregate_sales",
                   {"start_date": "2026-01-01", "end_date": "2026-01-31"})
    assert (a["total_paise"], a["count"], a["average_paise"]) == (15000, 2, 7500)
    assert b["total_paise"] == 99900


async def test_group_by_and_trend_zero_fill(mcp_url, settings, store, user_a):
    await seed(store, user_a, [("2026-01-01", 1000), ("2026-01-01", 500), ("2026-01-03", 2000)])
    t = rd(settings, user_a)
    grouped = await call(mcp_url, t, "aggregate_sales", {
        "start_date": "2026-01-01", "end_date": "2026-01-03", "group_by": "day"})
    assert [(r["label"], r["total_paise"], r["count"]) for r in grouped["rows"]] == [
        ("2026-01-01", 1500, 2), ("2026-01-03", 2000, 1)]
    trend = await call(mcp_url, t, "get_sales_trend", {
        "start_date": "2026-01-01", "end_date": "2026-01-03", "granularity": "day"})
    assert [p["total_paise"] for p in trend["points"]] == [1500, 0, 2000]
    assert trend["points"][1]["change_pct"] == -100.0 and trend["peak_label"] == "2026-01-03"


async def test_invalid_input_is_rejected(mcp_url, settings, user_a):
    t = rd(settings, user_a)
    with pytest.raises(ToolError):
        await call(mcp_url, t, "get_sales", {"date": "2026-13-45"})
    with pytest.raises(ToolError):
        await call(mcp_url, t, "aggregate_sales",
                   {"start_date": "2026-02-01", "end_date": "2026-01-01"})
    with pytest.raises(ToolError):
        await call(mcp_url, t, "preview_sales_selection", {"selector": {"date": "2026-01-01",
                   "start_date": "2026-01-01", "end_date": "2026-01-02"}})


async def test_writes_need_write_scope_bound_to_op(mcp_url, settings, user_a):
    args = {"date": "2026-01-01", "amount_paise": 10000, "op_key": "op-aaaaaaaa"}
    with pytest.raises(ToolError, match="Forbidden"):  # read token cannot write
        await call(mcp_url, rd(settings, user_a), "insert_sale", args)
    with pytest.raises(ToolError, match="not bound"):  # token for a different operation
        await call(mcp_url, wr(settings, user_a, "op-bbbbbbbb"), "insert_sale", args)
    with pytest.raises(ToolError):  # write token cannot read
        await call(mcp_url, wr(settings, user_a, "op-aaaaaaaa"), "get_sales", {"date": "2026-01-01"})


async def test_insert_is_idempotent_per_op_key(mcp_url, settings, user_a):
    args = {"date": "2026-01-01", "amount_paise": 10000, "op_key": "op-insert-1", "note": "cash"}
    first = await call(mcp_url, wr(settings, user_a, "op-insert-1"), "insert_sale", args)
    second = await call(mcp_url, wr(settings, user_a, "op-insert-1"), "insert_sale", args)
    assert first["replayed"] is False and second["replayed"] is True
    assert first["record"]["id"] == second["record"]["id"]
    listing = await call(mcp_url, rd(settings, user_a), "get_sales", {"date": "2026-01-01"})
    assert listing["total_count"] == 1 and listing["total_paise"] == 10000


async def test_update_and_delete_only_touch_own_data(mcp_url, settings, store, user_a, user_b):
    await seed(store, user_a, [("2026-01-01", 1000), ("2026-01-02", 2000)])
    await seed(store, user_b, [("2026-01-01", 7000)])
    sel = {"start_date": "2026-01-01", "end_date": "2026-01-31"}
    preview = await call(mcp_url, rd(settings, user_a), "preview_sales_selection", {"selector": sel})
    assert preview["count"] == 2 and preview["total_paise"] == 3000 and preview["span_days"] == 2
    upd = await call(mcp_url, wr(settings, user_a, "op-update-1"), "update_sales", {
        "selector": sel, "changes": {"note": "audited"}, "op_key": "op-update-1"})
    assert upd["affected"] == 2
    dele = await call(mcp_url, wr(settings, user_a, "op-delete-1"), "delete_sales",
                      {"selector": sel, "op_key": "op-delete-1"})
    assert dele["affected"] == 2 and dele["total_paise"] == 3000
    replay = await call(mcp_url, wr(settings, user_a, "op-delete-1"), "delete_sales",
                        {"selector": sel, "op_key": "op-delete-1"})
    assert replay["replayed"] is True and replay["affected"] == 2
    remaining = await call(mcp_url, rd(settings, user_b), "get_sales", {"date": "2026-01-01"})
    assert remaining["total_paise"] == 7000  # user B untouched


async def test_bulk_limit_and_range_new_date_guard(mcp_url, settings, store, user_a):
    await seed(store, user_a, [("2026-01-01", 100)] * 6)  # store fixture max_bulk=5
    sel = {"start_date": "2026-01-01", "end_date": "2026-01-31"}
    with pytest.raises(ToolError, match="max 5"):
        await call(mcp_url, wr(settings, user_a, "op-delete-2"), "delete_sales",
                   {"selector": sel, "op_key": "op-delete-2"})
    with pytest.raises(ToolError, match="new_date"):
        await call(mcp_url, wr(settings, user_a, "op-update-2"), "update_sales", {
            "selector": sel, "changes": {"new_date": "2026-02-01"}, "op_key": "op-update-2"})
    still = await call(mcp_url, rd(settings, user_a), "get_sales", {"date": "2026-01-01"})
    assert still["total_count"] == 6
