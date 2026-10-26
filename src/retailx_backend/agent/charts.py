"""Deterministic chart payload builder (structured JSON rendered by the React/Recharts UI)."""

from __future__ import annotations

from typing import Any

from .errors import StepError
from .plan import MAX_CHART_POINTS


def build_chart(step: dict[str, Any], source: dict[str, Any]) -> dict[str, Any] | None:
    """Return a chart spec, or ``None`` when the source period has no sales to show."""
    rows = source.get("points") or source.get("rows") or []
    if not rows:
        raise StepError("There is no data to chart for that selection.")
    data = [{"label": r["label"], "total": r["total"], "count": r.get("count", 0)}
            for r in rows[:MAX_CHART_POINTS]]
    if not any(point["total"] for point in data):
        return None
    return {
        "id": f"chart-{step['id']}", "type": step["chart_kind"], "title": step["chart_title"],
        "x_key": "label", "series": [{"key": "total", "label": "Sales (\u20b9)"}],
        "data": data, "currency": "INR",
    }
