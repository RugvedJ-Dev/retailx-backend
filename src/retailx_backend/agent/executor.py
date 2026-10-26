"""Executes one validated plan step. Reads and writes go through the MCP gateway only."""

from __future__ import annotations

from typing import Any

from .charts import build_chart
from .compute import compare_periods
from .deps import Deps
from .errors import StepError


def _data(results: dict[str, dict[str, Any]], step_id: str) -> dict[str, Any]:
    result = results.get(step_id)
    if not result or result.get("status") != "ok":
        raise StepError("A step this one depends on did not complete.")
    return result["data"]


async def execute_step(
    step: dict[str, Any], results: dict[str, dict[str, Any]], deps: Deps, op_key: str
) -> dict[str, Any]:
    a, mcp = step["action"], deps.mcp
    if a == "get_sales":
        return await mcp.read("get_sales", {"date": step["date"]})
    if a == "get_sales_range":
        return await mcp.read("get_sales_range", {"start_date": step["start_date"],
                                                  "end_date": step["end_date"]})
    if a == "aggregate_sales":
        return await mcp.read("aggregate_sales", {"start_date": step["start_date"],
                                                  "end_date": step["end_date"],
                                                  "group_by": step["group_by"]})
    if a == "sales_trend":
        return await mcp.read("get_sales_trend", {"start_date": step["start_date"],
                                                  "end_date": step["end_date"],
                                                  "granularity": step["granularity"]})
    if a == "compare_periods":
        cur, base = step["source_steps"]
        return compare_periods(_data(results, cur), _data(results, base))
    if a == "build_chart":
        return build_chart(step, _data(results, step["source_step"])) or {}
    if a == "insert_sale":
        return await mcp.write("insert_sale", {"date": step["date"],
                                               "amount_paise": step["amount_paise"],
                                               "note": step.get("note", "")}, op_key)
    if a == "update_sales":
        return await mcp.write("update_sales", {"selector": step["selector"],
                                                "changes": step["changes"]}, op_key)
    if a == "delete_sales":
        return await mcp.write("delete_sales", {"selector": step["selector"]}, op_key)
    raise StepError(f"Unsupported action {a!r}")
