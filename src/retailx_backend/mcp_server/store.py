"""MongoDB-backed sales store. Every method is scoped by ``user_id`` (tenant isolation).

Writes are idempotent per ``(user_id, op_key)``: a retried write returns the stored result and
never applies twice.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

from ..domain import (
    MAX_RANGE_DAYS,
    Granularity,
    iter_labels,
    parse_date,
    pct_change,
    period_label,
    rupees,
    validate_range,
)
from .models import (
    AggregateResult,
    AggregateRow,
    GroupBy,
    SaleChanges,
    SaleRecord,
    SalesList,
    SelectionPreview,
    Selector,
    TrendPoint,
    TrendResult,
    WriteResult,
)

_GROUP_FIELD = {"day": "date", "week": "week", "month": "month"}


def _derived(day: date) -> dict[str, str]:
    return {"date": day.isoformat(), "month": period_label(day, "month"),
            "week": period_label(day, "week")}


def _record(doc: dict) -> SaleRecord:
    created = doc.get("created_at")
    return SaleRecord(
        id=str(doc["_id"]), date=doc["date"], amount=rupees(doc["amount_paise"]),
        amount_paise=doc["amount_paise"], note=doc.get("note", ""),
        created_at=created.isoformat() if isinstance(created, datetime) else str(created or ""),
    )


class SalesStore:
    def __init__(self, db: Any | None = None, max_bulk: int = 500) -> None:
        self._db = db
        self.max_bulk = max_bulk
        self._ready = False
        self._ready_lock = asyncio.Lock()

    def bind(self, db: Any) -> None:
        self._db = db
        self._ready = False

    @property
    def sales(self) -> Any:
        if self._db is None:
            raise RuntimeError("SalesStore is not bound to a database")
        return self._db.sales

    @property
    def ops(self) -> Any:
        return self._db.write_ops

    async def ready(self) -> None:
        if self._ready:
            return
        async with self._ready_lock:
            if not self._ready:
                await self.ensure_indexes()
                self._ready = True

    async def ensure_indexes(self) -> None:
        await self.sales.create_index([("user_id", ASCENDING), ("date", ASCENDING)], name="user_date")
        await self.sales.create_index([("user_id", ASCENDING), ("month", ASCENDING)], name="user_month")
        await self.sales.create_index([("user_id", ASCENDING), ("week", ASCENDING)], name="user_week")
        await self.sales.create_index(
            [("user_id", ASCENDING), ("op_key", ASCENDING)], unique=True, name="user_op_key",
            partialFilterExpression={"op_key": {"$type": "string"}},
        )
        await self.ops.create_index(
            [("user_id", ASCENDING), ("op_key", ASCENDING)], unique=True, name="uniq_op"
        )
        await self.ops.create_index([("created_at", ASCENDING)], expireAfterSeconds=30 * 24 * 3600,
                                    name="op_ttl")

    # ------------------------------------------------------------------ helpers
    def _filter(self, user_id: str, sel: Selector) -> dict:
        flt: dict[str, Any] = {"user_id": user_id}
        if sel.mode == "ids":
            try:
                flt["_id"] = {"$in": [ObjectId(i) for i in sel.record_ids or []]}
            except (InvalidId, TypeError) as exc:
                raise ValueError("record_ids contains an invalid id") from exc
        elif sel.mode == "date":
            flt["date"] = sel.date
        else:
            flt["date"] = {"$gte": sel.start_date, "$lte": sel.end_date}
        return flt

    async def _summary(self, flt: dict) -> dict:
        cursor = await self.sales.aggregate([
            {"$match": flt},
            {"$group": {"_id": None, "count": {"$sum": 1}, "total": {"$sum": "$amount_paise"},
                        "min": {"$min": "$date"}, "max": {"$max": "$date"}}},
        ])
        rows = await cursor.to_list(length=1)
        if not rows:
            return {"count": 0, "total": 0, "min": None, "max": None}
        return rows[0]

    # -------------------------------------------------------------------- reads
    async def _list(self, user_id: str, flt: dict, limit: int) -> SalesList:
        summary = await self._summary(flt)
        cursor = self.sales.find(flt).sort([("date", 1), ("created_at", 1)]).limit(limit)
        docs = await cursor.to_list(length=limit)
        return SalesList(
            records=[_record(d) for d in docs], returned=len(docs), total_count=summary["count"],
            total_paise=summary["total"], total=rupees(summary["total"]),
            truncated=summary["count"] > len(docs),
        )

    async def sales_for_date(self, user_id: str, day: str, limit: int = 50) -> SalesList:
        parse_date(day)
        return await self._list(user_id, {"user_id": user_id, "date": day}, limit)

    async def sales_in_range(self, user_id: str, start: str, end: str, limit: int = 200) -> SalesList:
        validate_range(start, end)
        return await self._list(
            user_id, {"user_id": user_id, "date": {"$gte": start, "$lte": end}}, limit
        )

    async def _grouped(self, user_id: str, start: str, end: str, group_by: str) -> list[dict]:
        field = _GROUP_FIELD[group_by]
        cursor = await self.sales.aggregate([
            {"$match": {"user_id": user_id, "date": {"$gte": start, "$lte": end}}},
            {"$group": {"_id": f"${field}", "total": {"$sum": "$amount_paise"},
                        "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ])
        return await cursor.to_list(length=MAX_RANGE_DAYS + 5)

    async def aggregate(self, user_id: str, start: str, end: str, group_by: GroupBy) -> AggregateResult:
        validate_range(start, end)
        flt = {"user_id": user_id, "date": {"$gte": start, "$lte": end}}
        summary = await self._summary(flt)
        rows: list[AggregateRow] = []
        if group_by != "none":
            rows = [AggregateRow(label=r["_id"], total_paise=r["total"], total=rupees(r["total"]),
                                 count=r["count"]) for r in await self._grouped(user_id, start, end, group_by)]
        count, total = summary["count"], summary["total"]
        return AggregateResult(
            start_date=start, end_date=end, group_by=group_by, rows=rows, total_paise=total,
            total=rupees(total), count=count, average_paise=round(total / count) if count else 0,
        )

    async def trend(self, user_id: str, start: str, end: str, granularity: Granularity) -> TrendResult:
        s, e = validate_range(start, end)
        by_label = {r["_id"]: r for r in await self._grouped(user_id, start, end, granularity)}
        points: list[TrendPoint] = []
        previous: int | None = None
        for label in iter_labels(s, e, granularity):
            row = by_label.get(label, {"total": 0, "count": 0})
            change = pct_change(previous, row["total"]) if previous is not None else None
            points.append(TrendPoint(label=label, total_paise=row["total"], total=rupees(row["total"]),
                                     count=row["count"], change_pct=change))
            previous = row["total"]
        total = sum(p.total_paise for p in points)
        peak = max(points, key=lambda p: p.total_paise, default=None)
        low = min(points, key=lambda p: p.total_paise, default=None)
        return TrendResult(
            start_date=start, end_date=end, granularity=granularity, points=points,
            total_paise=total, total=rupees(total),
            overall_change_pct=pct_change(points[0].total_paise, points[-1].total_paise) if points else None,
            peak_label=peak.label if peak and peak.total_paise else None,
            lowest_label=low.label if low else None,
        )

    async def preview(self, user_id: str, sel: Selector) -> SelectionPreview:
        flt = self._filter(user_id, sel)
        summary = await self._summary(flt)
        docs = await self.sales.find(flt).sort([("date", 1), ("created_at", 1)]).limit(5).to_list(length=5)
        span = 0
        if summary["min"] and summary["max"]:
            span = (date.fromisoformat(summary["max"]) - date.fromisoformat(summary["min"])).days + 1
        return SelectionPreview(
            count=summary["count"], total_paise=summary["total"], total=rupees(summary["total"]),
            min_date=summary["min"], max_date=summary["max"], span_days=span,
            sample=[_record(d) for d in docs],
        )

    # ------------------------------------------------------------------- writes
    async def _idempotent(self, user_id: str, op_key: str, tool: str, run) -> WriteResult:
        existing = await self.ops.find_one({"user_id": user_id, "op_key": op_key})
        if existing and existing.get("status") == "done":
            return WriteResult(**{**existing["result"], "replayed": True})
        if not existing:
            try:
                await self.ops.insert_one({"user_id": user_id, "op_key": op_key, "tool": tool,
                                           "status": "started", "created_at": datetime.now(UTC)})
            except DuplicateKeyError:
                pass
        result: WriteResult = await run()
        await self.ops.update_one({"user_id": user_id, "op_key": op_key},
                                  {"$set": {"status": "done", "result": result.model_dump()}})
        return result

    async def insert_sale(self, user_id: str, day: str, amount_paise: int, note: str, op_key: str) -> WriteResult:
        d = parse_date(day)

        async def run() -> WriteResult:
            doc = {"user_id": user_id, "amount_paise": amount_paise, "note": note, "op_key": op_key,
                   "created_at": datetime.now(UTC), **_derived(d)}
            try:
                inserted = await self.sales.insert_one(doc)
                doc["_id"] = inserted.inserted_id
            except DuplicateKeyError:  # crash-after-insert retry: return the original record
                doc = await self.sales.find_one({"user_id": user_id, "op_key": op_key})
            return WriteResult(op_key=op_key, tool="insert_sale", affected=1,
                               total_paise=amount_paise, total=rupees(amount_paise), record=_record(doc))

        return await self._idempotent(user_id, op_key, "insert_sale", run)

    async def update_sales(self, user_id: str, sel: Selector, changes: SaleChanges, op_key: str) -> WriteResult:
        if changes.new_date and sel.mode == "range":
            raise ValueError("new_date cannot be applied to a date range; select by date or record_ids")
        flt = self._filter(user_id, sel)

        async def run() -> WriteResult:
            before = await self._summary(flt)
            if before["count"] > self.max_bulk:
                raise ValueError(f"Operation would affect {before['count']} records (max {self.max_bulk})")
            update: dict[str, Any] = {}
            if changes.amount_paise is not None:
                update["amount_paise"] = changes.amount_paise
            if changes.note is not None:
                update["note"] = changes.note
            if changes.new_date is not None:
                update.update(_derived(parse_date(changes.new_date)))
            result = await self.sales.update_many(flt, {"$set": update})
            after = await self._summary({**flt, **({"date": changes.new_date} if changes.new_date else {})})
            return WriteResult(op_key=op_key, tool="update_sales", affected=result.matched_count,
                               total_paise=after["total"], total=rupees(after["total"]))

        return await self._idempotent(user_id, op_key, "update_sales", run)

    async def delete_sales(self, user_id: str, sel: Selector, op_key: str) -> WriteResult:
        flt = self._filter(user_id, sel)

        async def run() -> WriteResult:
            before = await self._summary(flt)
            if before["count"] > self.max_bulk:
                raise ValueError(f"Operation would affect {before['count']} records (max {self.max_bulk})")
            result = await self.sales.delete_many(flt)
            return WriteResult(op_key=op_key, tool="delete_sales", affected=result.deleted_count,
                               total_paise=before["total"], total=rupees(before["total"]))

        return await self._idempotent(user_id, op_key, "delete_sales", run)
