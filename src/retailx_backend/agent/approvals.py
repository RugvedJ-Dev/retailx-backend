"""Human-in-the-loop approval payloads (what the confirmation dialog shows)."""

from __future__ import annotations

import uuid
from typing import Any

from ..domain import fmt_money

_APPROVAL_NS = uuid.UUID("6f1b0a3e-5b0a-4a5e-9f0e-2d2f6b1c9a11")
LARGE_SPAN_DAYS = 31
LARGE_COUNT = 50


def approval_id_for(run_id: str, step_id: str) -> str:
    """Deterministic id: LangGraph re-runs the gate node on resume and must derive the same id."""
    return str(uuid.uuid5(_APPROVAL_NS, f"{run_id}:{step_id}"))


def selector_text(sel: dict[str, str]) -> str:
    return f"on {sel['date']}" if "date" in sel else f"from {sel['start_date']} to {sel['end_date']}"


def _changes_text(changes: dict[str, Any]) -> str:
    parts = []
    if "amount_paise" in changes:
        parts.append(f"set amount to {fmt_money(changes['amount_paise'])}")
    if "note" in changes:
        parts.append(f"set note to \"{changes['note']}\"")
    if "new_date" in changes:
        parts.append(f"move to {changes['new_date']}")
    return ", ".join(parts)


def build_payload(step: dict[str, Any], preview: dict[str, Any] | None, approval_id: str) -> dict[str, Any]:
    action = step["action"]
    details: dict[str, Any] = {}
    warnings: list[str] = []
    if action == "insert_sale":
        summary = f"Add a sale of {fmt_money(step['amount_paise'])} on {step['date']}"
        details = {"date": step["date"], "amount": fmt_money(step["amount_paise"]),
                   "note": step.get("note", "")}
    else:
        assert preview is not None
        count, total = preview["count"], fmt_money(preview["total_paise"])
        where = selector_text(step["selector"])
        if action == "delete_sales":
            summary = f"Permanently delete {count} sales record(s) {where}, totalling {total}"
        else:
            summary = (f"Update {count} sales record(s) {where} (currently totalling {total}): "
                       f"{_changes_text(step['changes'])}")
            details["changes"] = _changes_text(step["changes"])
        details.update(records=count, total=total, date_span=f"{preview['min_date']} to {preview['max_date']}",
                       sample=[{"date": r["date"], "amount": fmt_money(r["amount_paise"]),
                                "note": r["note"]} for r in preview["sample"]])
        if preview["span_days"] > LARGE_SPAN_DAYS:
            warnings.append(f"This affects records spread over {preview['span_days']} days.")
        if count >= LARGE_COUNT:
            warnings.append(f"This affects {count} records.")
    return {"kind": "approval", "approval_id": approval_id, "step_id": step["id"],
            "operation": action, "destructive": action == "delete_sales", "summary": summary,
            "details": details, "warnings": warnings}
