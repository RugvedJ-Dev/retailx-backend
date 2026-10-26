"""MCP client gateway: per-user short-lived tokens, timeouts, retries and a user-isolated cache."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastmcp import Client
from fastmcp.exceptions import ToolError

from ..config import Settings
from ..security import SCOPE_READ, SCOPE_WRITE, mint_mcp_token
from .errors import ToolFailure

log = logging.getLogger("retailx.mcp")
MAX_ENTRIES_PER_USER = 200


class ReadCache:
    """Short-TTL cache of read results, partitioned by user and cleared on that user's writes."""

    def __init__(self, ttl_seconds: float) -> None:
        self.ttl = ttl_seconds
        self._data: dict[str, dict[str, tuple[float, dict]]] = {}

    def get(self, user_id: str, key: str) -> dict | None:
        entry = self._data.get(user_id, {}).get(key)
        if entry and entry[0] > time.monotonic():
            return entry[1]
        return None

    def put(self, user_id: str, key: str, value: dict) -> None:
        if self.ttl <= 0:
            return
        bucket = self._data.setdefault(user_id, {})
        if len(bucket) >= MAX_ENTRIES_PER_USER:
            bucket.clear()
        bucket[key] = (time.monotonic() + self.ttl, value)

    def invalidate(self, user_id: str) -> None:
        self._data.pop(user_id, None)


class McpGateway:
    def __init__(self, settings: Settings, user_id: str, cache: ReadCache) -> None:
        self.settings, self.user_id, self.cache = settings, user_id, cache

    async def _call(self, token: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(2):
            try:
                async with asyncio.timeout(self.settings.mcp_timeout_seconds):
                    async with Client(self.settings.mcp_url, auth=token) as client:
                        result = await client.call_tool(tool, args)
                return dict(result.structured_content or {})
            except ToolError as exc:  # validation / authorization: retrying cannot help
                raise ToolFailure(str(exc)) from exc
            except Exception as exc:  # transport problems, timeouts
                last = exc
                log.warning("mcp call failed tool=%s attempt=%d err=%r", tool, attempt + 1, exc)
                await asyncio.sleep(0.3 * (attempt + 1))
        raise ToolFailure("The sales service is unavailable right now. Please try again.") from last

    async def read(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        key = f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"
        cached = self.cache.get(self.user_id, key)
        if cached is not None:
            return cached
        token = mint_mcp_token(self.settings, self.user_id, [SCOPE_READ])
        data = await self._call(token, tool, args)
        self.cache.put(self.user_id, key, data)
        return data

    async def write(self, tool: str, args: dict[str, Any], op_key: str) -> dict[str, Any]:
        """Write with a token bound to ``op_key``; only called after human approval."""
        token = mint_mcp_token(self.settings, self.user_id, [SCOPE_WRITE], op_key=op_key)
        try:
            return await self._call(token, tool, {**args, "op_key": op_key})
        finally:
            self.cache.invalidate(self.user_id)
