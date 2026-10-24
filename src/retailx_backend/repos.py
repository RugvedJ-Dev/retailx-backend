"""MongoDB repositories for API-owned collections: users, threads, messages, approvals.

Every thread/message/approval accessor is scoped by ``user_id`` (tenant isolation).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

RESOLVING_RECLAIM_SECONDS = 300


class EmailTaken(Exception):
    pass


def now() -> datetime:
    return datetime.now(UTC)


def to_oid(value: str | ObjectId) -> ObjectId | None:
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


# ----------------------------------------------------------------------------- users
async def create_user(db: Any, *, name: str, email: str, password_hash: str) -> dict:
    doc = {"name": name.strip(), "email": email.lower(), "password_hash": password_hash,
           "created_at": now()}
    try:
        result = await db.users.insert_one(doc)
    except DuplicateKeyError as exc:
        raise EmailTaken from exc
    doc["_id"] = result.inserted_id
    return doc


async def get_user_by_email(db: Any, email: str) -> dict | None:
    return await db.users.find_one({"email": email.lower()})


async def get_user(db: Any, user_id: str) -> dict | None:
    oid = to_oid(user_id)
    return await db.users.find_one({"_id": oid}) if oid else None


# --------------------------------------------------------------------------- threads
async def create_thread(db: Any, user_id: ObjectId, name: str) -> dict:
    ts = now()
    doc = {"user_id": user_id, "name": name.strip(), "created_at": ts, "updated_at": ts,
           "run_lock_until": None}
    result = await db.threads.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def list_threads(db: Any, user_id: ObjectId, limit: int = 200) -> list[dict]:
    cursor = db.threads.find({"user_id": user_id}).sort("updated_at", -1).limit(limit)
    return await cursor.to_list(length=limit)


async def get_thread(db: Any, user_id: ObjectId, thread_id: str) -> dict | None:
    oid = to_oid(thread_id)
    if oid is None:
        return None
    return await db.threads.find_one({"_id": oid, "user_id": user_id})


async def delete_thread(db: Any, user_id: ObjectId, thread_id: str) -> bool:
    thread = await get_thread(db, user_id, thread_id)
    if thread is None:
        return False
    await db.messages.delete_many({"thread_id": thread["_id"], "user_id": user_id})
    await db.approvals.delete_many({"thread_id": thread["_id"], "user_id": user_id})
    await db.threads.delete_one({"_id": thread["_id"], "user_id": user_id})
    return True


async def touch_thread(db: Any, thread_oid: ObjectId) -> None:
    await db.threads.update_one({"_id": thread_oid}, {"$set": {"updated_at": now()}})


async def acquire_run_lock(db: Any, thread_oid: ObjectId, user_id: ObjectId, ttl: int) -> bool:
    """Lease-based lock so only one agent run executes per thread (works across processes)."""
    ts = now()
    result = await db.threads.update_one(
        {"_id": thread_oid, "user_id": user_id,
         "$or": [{"run_lock_until": None}, {"run_lock_until": {"$lt": ts}}]},
        {"$set": {"run_lock_until": ts + timedelta(seconds=ttl)}},
    )
    return result.matched_count == 1


async def release_run_lock(db: Any, thread_oid: ObjectId) -> None:
    await db.threads.update_one({"_id": thread_oid}, {"$set": {"run_lock_until": None}})


# -------------------------------------------------------------------------- messages
async def add_message(
    db: Any, *, thread_oid: ObjectId, user_id: ObjectId, role: str, content: str,
    charts: list[dict] | None = None,
) -> dict:
    doc = {"thread_id": thread_oid, "user_id": user_id, "role": role, "content": content,
           "charts": charts or [], "created_at": now()}
    result = await db.messages.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def list_messages(db: Any, thread_oid: ObjectId, user_id: ObjectId, limit: int = 500) -> list[dict]:
    cursor = db.messages.find({"thread_id": thread_oid, "user_id": user_id}).sort("created_at", 1)
    return await cursor.limit(limit).to_list(length=limit)


# ------------------------------------------------------------------------- approvals
async def upsert_approval(
    db: Any, *, approval_id: str, thread_oid: ObjectId, user_id: ObjectId, run_id: str,
    payload: dict, ttl_hours: int,
) -> dict:
    ts = now()
    doc = {"thread_id": thread_oid, "user_id": user_id, "run_id": run_id, "payload": payload,
           "status": "pending", "decision": None, "created_at": ts,
           "expires_at": ts + timedelta(hours=ttl_hours)}
    await db.approvals.update_one({"_id": approval_id}, {"$setOnInsert": doc}, upsert=True)
    return await db.approvals.find_one({"_id": approval_id})


async def get_approval(db: Any, approval_id: str, user_id: ObjectId) -> dict | None:
    return await db.approvals.find_one({"_id": approval_id, "user_id": user_id})


async def get_pending_approval(db: Any, thread_oid: ObjectId, user_id: ObjectId) -> dict | None:
    return await db.approvals.find_one(
        {"thread_id": thread_oid, "user_id": user_id, "status": {"$in": ["pending", "resolving"]},
         "expires_at": {"$gt": now()}}
    )


async def claim_approval(
    db: Any, *, approval_id: str, thread_oid: ObjectId, user_id: ObjectId, decision: str
) -> dict | None:
    """Atomically move an approval to ``resolving`` so the graph is resumed at most once.

    A ``resolving`` claim older than ``RESOLVING_RECLAIM_SECONDS`` (crashed worker) may be
    re-claimed; the write itself is protected by its idempotency key.
    """
    ts = now()
    return await db.approvals.find_one_and_update(
        {"_id": approval_id, "thread_id": thread_oid, "user_id": user_id,
         "expires_at": {"$gt": ts},
         "$or": [{"status": "pending"},
                 {"status": "resolving",
                  "claimed_at": {"$lt": ts - timedelta(seconds=RESOLVING_RECLAIM_SECONDS)}}]},
        {"$set": {"status": "resolving", "decision": decision, "claimed_at": ts}},
        return_document=ReturnDocument.AFTER,
    )


async def finish_approval(db: Any, approval_id: str, status: str) -> None:
    await db.approvals.update_one(
        {"_id": approval_id}, {"$set": {"status": status, "resolved_at": now()}}
    )
