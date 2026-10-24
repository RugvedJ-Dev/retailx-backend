"""Pure helpers shared by the API/agent and the MCP server (no I/O, no framework imports)."""

from __future__ import annotations

import re
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Literal

Granularity = Literal["day", "week", "month"]

MIN_DATE = date(2000, 1, 1)
MAX_DATE = date(2100, 12, 31)
MAX_AMOUNT_PAISE = 10_000_000_000  # Rs 10 crore per record
MAX_RANGE_DAYS = 1100
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_date(value: str) -> date:
    """Strictly parse ``YYYY-MM-DD``."""
    if not isinstance(value, str) or not _ISO_DATE.match(value):
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date {value!r}") from exc
    if not MIN_DATE <= parsed <= MAX_DATE:
        raise ValueError(f"Date {value} is outside the supported range")
    return parsed


def validate_range(start: str, end: str, max_days: int = MAX_RANGE_DAYS) -> tuple[date, date]:
    s, e = parse_date(start), parse_date(end)
    if e < s:
        raise ValueError("end_date must not be before start_date")
    if (e - s).days + 1 > max_days:
        raise ValueError(f"Date range too large (max {max_days} days)")
    return s, e


def to_paise(amount: float | int | str | Decimal) -> int:
    """Convert a rupee amount to integer paise using exact decimal arithmetic."""
    try:
        value = Decimal(str(amount))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid amount {amount!r}") from exc
    if not value.is_finite():
        raise ValueError("Amount must be a finite number")
    return int((value * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def rupees(paise: int) -> float:
    return round(paise / 100, 2)


def fmt_money(paise: int) -> str:
    """Format paise as Indian-grouped rupees, e.g. Rs 1,23,456.50 -> \u20b91,23,456.50."""
    sign = "-" if paise < 0 else ""
    whole, frac = divmod(abs(int(paise)), 100)
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
        digits = f"{head},{tail}"
    return f"{sign}\u20b9{digits}.{frac:02d}"


def period_label(day: date, granularity: Granularity) -> str:
    if granularity == "day":
        return day.isoformat()
    if granularity == "month":
        return f"{day.year:04d}-{day.month:02d}"
    iso = day.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def iter_labels(start: date, end: date, granularity: Granularity) -> list[str]:
    """All period labels covering [start, end], in chronological order, without duplicates."""
    labels: list[str] = []
    seen: set[str] = set()
    current = start
    while current <= end:
        label = period_label(current, granularity)
        if label not in seen:
            seen.add(label)
            labels.append(label)
        current += timedelta(days=1)
    return labels


def pct_change(old: int, new: int) -> float | None:
    """Percentage change from ``old`` to ``new``; ``None`` when undefined (old == 0)."""
    if old == 0:
        return None
    return round((new - old) / old * 100, 2)
