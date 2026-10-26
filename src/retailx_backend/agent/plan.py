"""Planner output schema, deterministic plan validation and the approval policy.

The LLM only fills a closed, typed schema. ``validate_plan`` re-checks every field in code and
normalises it into the arguments the executor uses, so nothing the model wrote is ever executed
as free-form instructions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..domain import MAX_AMOUNT_PAISE, iter_labels, parse_date, to_paise, validate_range
from .errors import PlanError

Action = Literal[
    "get_sales", "get_sales_range", "aggregate_sales", "sales_trend", "compare_periods",
    "build_chart", "insert_sale", "update_sales", "delete_sales",
]
READ_ACTIONS = frozenset({"get_sales", "get_sales_range", "aggregate_sales", "sales_trend"})
WRITE_ACTIONS = frozenset({"insert_sale", "update_sales", "delete_sales"})
MAX_CHART_POINTS = 400


class PlanStep(BaseModel):
    id: str = Field(description="s1, s2, ... in execution order")
    title: str = Field(description="Short user-facing description of this step")
    action: Action
    date: str | None = Field(default=None, description="YYYY-MM-DD")
    start_date: str | None = Field(default=None, description="YYYY-MM-DD")
    end_date: str | None = Field(default=None, description="YYYY-MM-DD, inclusive")
    group_by: Literal["none", "day", "week", "month"] | None = None
    granularity: Literal["day", "week", "month"] | None = None
    amount: float | None = Field(default=None, description="Rupees (insert_sale)")
    note: str | None = None
    new_amount: float | None = Field(default=None, description="Rupees (update_sales)")
    new_note: str | None = None
    new_date: str | None = Field(default=None, description="YYYY-MM-DD (update_sales)")
    source_steps: list[str] | None = None
    chart_kind: Literal["line", "bar", "pie"] | None = None
    chart_title: str | None = None


class Plan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    direct_reply: str | None = Field(
        default=None,
        description="Reply to give instead of steps (greeting, out of scope, or a clarifying question)",
    )


def _selector(s: PlanStep) -> dict[str, str]:
    has_range = s.start_date is not None or s.end_date is not None
    if s.date and not has_range:
        parse_date(s.date)
        return {"date": s.date}
    if has_range and not s.date:
        if not (s.start_date and s.end_date):
            raise ValueError("start_date and end_date are both required")
        validate_range(s.start_date, s.end_date)
        return {"start_date": s.start_date, "end_date": s.end_date}
    raise ValueError("provide either date or start_date+end_date")


def _source(s: PlanStep, prior: dict[str, PlanStep], count: int) -> list[str]:
    sources = s.source_steps or []
    if len(sources) != count:
        raise ValueError(f"source_steps must list exactly {count} earlier step id(s)")
    for sid in sources:
        if sid not in prior:
            raise ValueError(f"source step {sid!r} does not exist or is not earlier")
    return sources


def _validate_step(s: PlanStep, prior: dict[str, PlanStep]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": s.id, "title": s.title.strip()[:100] or s.action, "action": s.action}
    a = s.action
    if a == "get_sales":
        parse_date(s.date or "")
        out["date"] = s.date
    elif a in ("get_sales_range", "aggregate_sales", "sales_trend"):
        if not (s.start_date and s.end_date):
            raise ValueError("start_date and end_date are required")
        start, end = validate_range(s.start_date, s.end_date)
        out.update(start_date=s.start_date, end_date=s.end_date)
        if a == "aggregate_sales":
            out["group_by"] = s.group_by or "none"
        if a == "sales_trend":
            out["granularity"] = s.granularity or "day"
            if len(iter_labels(start, end, out["granularity"])) > MAX_CHART_POINTS:
                raise ValueError("too many points; use week or month granularity")
    elif a == "compare_periods":
        sources = _source(s, prior, 2)
        if sources[0] == sources[1] or any(prior[x].action != "aggregate_sales" for x in sources):
            raise ValueError("compare_periods needs two different aggregate_sales steps")
        out["source_steps"] = sources
    elif a == "build_chart":
        (src,) = _source(s, prior, 1)
        source = prior[src]
        grouped = source.action == "aggregate_sales" and (source.group_by or "none") != "none"
        if not (grouped or source.action in ("sales_trend", "compare_periods")):
            raise ValueError("build_chart needs a sales_trend, grouped aggregate_sales or "
                             "compare_periods source")
        default = "bar" if source.action == "compare_periods" else "line"
        out.update(source_step=src, chart_kind=s.chart_kind or default,
                   chart_title=(s.chart_title or out["title"]).strip()[:100])
    elif a == "insert_sale":
        parse_date(s.date or "")
        if s.amount is None:
            raise ValueError("amount is required")
        paise = to_paise(s.amount)
        if not 0 < paise <= MAX_AMOUNT_PAISE:
            raise ValueError("amount must be positive and within limits")
        out.update(date=s.date, amount_paise=paise, note=(s.note or "")[:200])
    elif a == "update_sales":
        out["selector"] = sel = _selector(s)
        changes: dict[str, Any] = {}
        if s.new_amount is not None:
            paise = to_paise(s.new_amount)
            if not 0 < paise <= MAX_AMOUNT_PAISE:
                raise ValueError("new_amount must be positive and within limits")
            changes["amount_paise"] = paise
        if s.new_note is not None:
            changes["note"] = s.new_note[:200]
        if s.new_date is not None:
            parse_date(s.new_date)
            if "start_date" in sel:
                raise ValueError("new_date cannot be applied to a date range")
            changes["new_date"] = s.new_date
        if not changes:
            raise ValueError("nothing to change")
        out["changes"] = changes
    elif a == "delete_sales":
        out["selector"] = _selector(s)
    return out


def validate_plan(plan: Plan, max_steps: int) -> list[dict[str, Any]]:
    """Return normalised steps or raise :class:`PlanError` listing every problem found."""
    errors: list[str] = []
    if len(plan.steps) > max_steps:
        errors.append(f"at most {max_steps} steps are allowed")
    prior: dict[str, PlanStep] = {}
    out: list[dict[str, Any]] = []
    for idx, step in enumerate(plan.steps[:max_steps]):
        if step.id != f"s{idx + 1}":
            errors.append(f"step ids must be s1..sN in order (got {step.id!r} at position {idx + 1})")
            continue
        try:
            out.append(_validate_step(step, prior))
        except ValueError as exc:
            errors.append(f"{step.id}: {exc}")
            continue
        prior[step.id] = step
    if errors:
        raise PlanError(errors)
    return out


def requires_approval(step: dict[str, Any], policy: str) -> bool:
    """Deterministic HITL policy. The LLM never decides whether approval is needed."""
    action = step["action"]
    if action not in WRITE_ACTIONS:
        return False
    if policy == "all_writes":
        return True
    return action in ("update_sales", "delete_sales")
