"""Deterministic computation: period comparison, result summaries and LLM-safe fact compaction."""

from __future__ import annotations

import calendar
from datetime import date
from typing import Any

from ..domain import fmt_money, pct_change, rupees


def range_label(start: str, end: str) -> str:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    if s == e:
        return start
    if s.day == 1 and (s.year, s.month) == (e.year, e.month) and e.day == calendar.monthrange(
        e.year, e.month)[1]:
        return s.strftime("%B %Y")
    return f"{start} to {end}"


def _period(agg: dict[str, Any]) -> dict[str, Any]:
    return {"label": range_label(agg["start_date"], agg["end_date"]), "start_date": agg["start_date"],
            "end_date": agg["end_date"], "total_paise": agg["total_paise"],
            "total": rupees(agg["total_paise"]), "count": agg["count"]}


def compare_periods(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Compare two aggregate results. All arithmetic is done here, never by the LLM."""
    cur, base = _period(current), _period(baseline)
    diff = cur["total_paise"] - base["total_paise"]
    return {
        "current": cur, "baseline": base, "diff_paise": diff, "diff": rupees(diff),
        "pct_change": pct_change(base["total_paise"], cur["total_paise"]),
        "direction": "up" if diff > 0 else "down" if diff < 0 else "flat",
        "rows": [base, cur],  # chronological order for charts
    }


def step_summary(step: dict[str, Any], data: dict[str, Any]) -> str:
    """One short, user-facing line describing a completed step."""
    a = step["action"]
    if data.get("skipped"):
        return "No matching records - nothing to change"
    if a in ("get_sales", "get_sales_range"):
        return f"{data['total_count']} sale(s) totalling {fmt_money(data['total_paise'])}"
    if a == "aggregate_sales":
        return f"{fmt_money(data['total_paise'])} across {data['count']} sale(s)"
    if a == "sales_trend":
        return f"{len(data['points'])} period(s), total {fmt_money(data['total_paise'])}"
    if a == "compare_periods":
        pct = data["pct_change"]
        return f"{data['direction']} {abs(pct)}%" if pct is not None else f"{data['direction']} vs baseline"
    if a == "build_chart":
        return "Chart ready" if data.get("type") else "Nothing to chart for this period"
    if a == "insert_sale":
        return f"Added {fmt_money(data['total_paise'])}"
    if a == "update_sales":
        return f"Updated {data['affected']} record(s)"
    if a == "delete_sales":
        return f"Deleted {data['affected']} record(s)"
    return "Done"


def compact_result(step: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Trim a step result to the facts the synthesis model needs (amounts pre-formatted)."""
    a = step["action"]
    if a in ("get_sales", "get_sales_range"):
        return {"total": fmt_money(data["total_paise"]), "total_records": data["total_count"],
                "records_shown": [{"date": r["date"], "amount": fmt_money(r["amount_paise"]),
                                   "note": r["note"]} for r in data["records"][:25]],
                "truncated": data["truncated"]}
    if a == "aggregate_sales":
        return {"start_date": data["start_date"], "end_date": data["end_date"],
                "total": fmt_money(data["total_paise"]), "count": data["count"],
                "average": fmt_money(data["average_paise"]),
                "groups": [{"label": r["label"], "total": fmt_money(r["total_paise"]),
                            "count": r["count"]} for r in data["rows"][:40]]}
    if a == "sales_trend":
        return {"granularity": data["granularity"], "total": fmt_money(data["total_paise"]),
                "overall_change_pct": data["overall_change_pct"], "peak": data["peak_label"],
                "lowest": data["lowest_label"],
                "points": [{"label": p["label"], "total": fmt_money(p["total_paise"]),
                            "change_pct": p["change_pct"]} for p in data["points"][:60]]}
    if a == "compare_periods":
        return {"current": {"label": data["current"]["label"],
                            "total": fmt_money(data["current"]["total_paise"])},
                "baseline": {"label": data["baseline"]["label"],
                             "total": fmt_money(data["baseline"]["total_paise"])},
                "difference": fmt_money(data["diff_paise"]), "pct_change": data["pct_change"],
                "direction": data["direction"]}
    if a == "build_chart":
        return {"chart_built": bool(data.get("type")), "title": data.get("title")}
    return {"affected_records": data.get("affected"), "total": fmt_money(data.get("total_paise", 0))}
