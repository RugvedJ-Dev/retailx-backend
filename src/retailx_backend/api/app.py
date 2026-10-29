"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..agent.graph import build_graph
from ..agent.llm import LangChainGateway, LLMGateway
from ..agent.runner import AgentRunner
from ..config import Settings, get_settings
from ..db import create_client, ensure_indexes
from .ratelimit import FailureLimiter
from .routes_auth import router as auth_router
from .routes_threads import router as threads_router

log = logging.getLogger("retailx.api")


def create_app(
    settings: Settings | None = None, *, db: Any | None = None, checkpointer: Any | None = None,
    llm: LLMGateway | None = None,
) -> FastAPI:
    """Build the app. ``db``/``checkpointer``/``llm`` can be injected (tests); otherwise they are
    created from settings (MongoDB + MongoDB checkpointer + LangChain gateway)."""
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = sync_client = None
        if db is None:
            client = create_client(settings.mongodb_uri)
            app.state.db = client[settings.mongodb_db]
        else:
            app.state.db = db
        await ensure_indexes(app.state.db)
        saver = checkpointer
        if saver is None:
            from langgraph.checkpoint.mongodb import MongoDBSaver
            from pymongo import MongoClient

            sync_client = MongoClient(settings.mongodb_uri, tz_aware=True)
            saver = MongoDBSaver(sync_client, db_name=settings.mongodb_db)
        app.state.runner = AgentRunner(settings=settings, db=app.state.db, graph=build_graph(saver),
                                       llm=llm or LangChainGateway(settings))
        yield
        if client is not None:
            await client.close()
        if sync_client is not None:
            sync_client.close()

    app = FastAPI(title="RetailX AI API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.login_limiter = FailureLimiter()
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"], allow_credentials=False,
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error request_id=%s path=%s", request_id, request.url.path)
            response = JSONResponse({"detail": "Internal server error"}, status_code=500)
        response.headers["X-Request-ID"] = request_id
        log.info("request id=%s method=%s path=%s status=%s ms=%d", request_id, request.method,
                 request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
        return response

    @app.get("/api/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(auth_router)
    app.include_router(threads_router)
    return app
