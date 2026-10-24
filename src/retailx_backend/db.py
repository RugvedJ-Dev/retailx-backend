"""MongoDB connection helpers and index management for API-owned collections.

The ``sales`` collection is owned by the MCP server (see ``mcp_server.store``); the API
and agent never read or write it directly.
"""

from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient


def create_client(uri: str) -> AsyncMongoClient:
    return AsyncMongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=5000)


async def ensure_indexes(db: Any) -> None:
    await db.users.create_index([("email", ASCENDING)], unique=True, name="uniq_email")
    await db.threads.create_index(
        [("user_id", ASCENDING), ("updated_at", DESCENDING)], name="user_threads"
    )
    await db.messages.create_index(
        [("thread_id", ASCENDING), ("created_at", ASCENDING)], name="thread_messages"
    )
    await db.approvals.create_index(
        [("thread_id", ASCENDING), ("status", ASCENDING)], name="thread_approvals"
    )
    # Housekeeping: drop approval records a week after they expire.
    await db.approvals.create_index(
        [("expires_at", ASCENDING)], expireAfterSeconds=7 * 24 * 3600, name="approval_ttl"
    )
