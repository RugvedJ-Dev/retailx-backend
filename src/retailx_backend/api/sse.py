"""Server-Sent Events plumbing.

The agent run is pumped by a background task that is independent of the HTTP connection, so a
client disconnect (closed tab, flaky mobile network) can never abandon a run half-way - for
example after an approved write was started. Results are persisted and visible on reload.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi.responses import StreamingResponse

log = logging.getLogger("retailx.sse")
HEARTBEAT_SECONDS = 15
_background: set[asyncio.Task] = set()


def format_sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"


def sse_response(
    events: AsyncIterator[dict], cleanup: Callable[[], Awaitable[None]]
) -> StreamingResponse:
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for event in events:
                await queue.put(event)
        except Exception:
            log.exception("unhandled error while streaming")
            await queue.put({"type": "error", "code": "internal", "message": "Internal error"})
        finally:
            try:
                await cleanup()
            finally:
                await queue.put({"type": "done"})
                await queue.put(None)

    task = asyncio.create_task(pump())
    _background.add(task)
    task.add_done_callback(_background.discard)

    async def body() -> AsyncIterator[str]:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if event is None:
                return
            yield format_sse(event)

    return StreamingResponse(
        body(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
