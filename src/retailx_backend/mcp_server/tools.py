"""MCP tool definitions (thin, validated wrappers over :class:`SalesStore`)."""

import functools
from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from .auth import SCOPE_READ, SCOPE_WRITE, current_caller
from .models import (
    AggregateResult,
    GroupBy,
    SaleChanges,
    SalesList,
    SelectionPreview,
    Selector,
    TrendResult,
    WriteResult,
)
from .store import SalesStore

DateStr = Annotated[str, Field(description="Date as YYYY-MM-DD", pattern=r"^\d{4}-\d{2}-\d{2}$")]
OpKey = Annotated[
    str,
    Field(description="Idempotency key bound to the approved operation",
          pattern=r"^[A-Za-z0-9:_.\-]{8,200}$"),
]

READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True}
WRITE = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True}
DESTRUCTIVE = {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True}


def _guarded(fn):
    """Translate domain validation errors into MCP tool errors."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ToolError:
            raise
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


def register_tools(mcp: FastMCP, store: SalesStore) -> None:
    async def _read_user() -> str:
        caller = current_caller()
        caller.require(SCOPE_READ)
        await store.ready()
        return caller.user_id

    async def _write_user(op_key: str) -> str:
        caller = current_caller()
        caller.require(SCOPE_WRITE, op_key)
        await store.ready()
        return caller.user_id

    @mcp.tool(name="get_sales", annotations=READ_ONLY,
              description="List the authenticated user's sales records for a single day.")
    @_guarded
    async def get_sales(
        date: DateStr, limit: Annotated[int, Field(ge=1, le=100)] = 50
    ) -> SalesList:
        return await store.sales_for_date(await _read_user(), date, limit)

    @mcp.tool(name="get_sales_range", annotations=READ_ONLY,
              description="List sales records between two dates (inclusive, max 1100 days).")
    @_guarded
    async def get_sales_range(
        start_date: DateStr, end_date: DateStr,
        limit: Annotated[int, Field(ge=1, le=200)] = 200,
    ) -> SalesList:
        return await store.sales_in_range(await _read_user(), start_date, end_date, limit)

    @mcp.tool(name="aggregate_sales", annotations=READ_ONLY,
              description="Total, count and average of sales in a date range, optionally grouped "
                          "by day, ISO week or month.")
    @_guarded
    async def aggregate_sales(
        start_date: DateStr, end_date: DateStr, group_by: GroupBy = "none"
    ) -> AggregateResult:
        return await store.aggregate(await _read_user(), start_date, end_date, group_by)

    @mcp.tool(name="get_sales_trend", annotations=READ_ONLY,
              description="Historical sales time series with zero-filled periods and "
                          "period-over-period change.")
    @_guarded
    async def get_sales_trend(
        start_date: DateStr, end_date: DateStr,
        granularity: Literal["day", "week", "month"] = "day",
    ) -> TrendResult:
        return await store.trend(await _read_user(), start_date, end_date, granularity)

    @mcp.tool(name="preview_sales_selection", annotations=READ_ONLY,
              description="Read-only preview (count, total, sample) of the records a selector "
                          "would affect. Used to build approval requests.")
    @_guarded
    async def preview_sales_selection(selector: Selector) -> SelectionPreview:
        return await store.preview(await _read_user(), selector)

    @mcp.tool(name="insert_sale", annotations=WRITE,
              description="Insert one sales record. Requires a write token bound to op_key.")
    @_guarded
    async def insert_sale(
        date: DateStr,
        amount_paise: Annotated[int, Field(gt=0, le=10_000_000_000)],
        op_key: OpKey,
        note: Annotated[str, Field(max_length=200)] = "",
    ) -> WriteResult:
        return await store.insert_sale(await _write_user(op_key), date, amount_paise, note, op_key)

    @mcp.tool(name="update_sales", annotations=WRITE,
              description="Update amount/note/date on selected records (max 500 per call). "
                          "Requires a write token bound to op_key.")
    @_guarded
    async def update_sales(selector: Selector, changes: SaleChanges, op_key: OpKey) -> WriteResult:
        return await store.update_sales(await _write_user(op_key), selector, changes, op_key)

    @mcp.tool(name="delete_sales", annotations=DESTRUCTIVE,
              description="Delete selected records (max 500 per call). "
                          "Requires a write token bound to op_key.")
    @_guarded
    async def delete_sales(selector: Selector, op_key: OpKey) -> WriteResult:
        return await store.delete_sales(await _write_user(op_key), selector, op_key)
