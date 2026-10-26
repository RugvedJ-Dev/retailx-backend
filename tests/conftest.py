import asyncio
import socket

import pytest
import uvicorn
from bson import ObjectId

from retailx_backend.config import Settings
from retailx_backend.mcp_server.server import create_server
from retailx_backend.mcp_server.settings import McpSettings
from retailx_backend.mcp_server.store import SalesStore
from tests.fakes import FakeDB

API_SECRET = "a" * 40
MCP_SECRET = "m" * 40


@pytest.fixture
def settings() -> Settings:
    return Settings(env="test", jwt_secret=API_SECRET, mcp_jwt_secret=MCP_SECRET)  # type: ignore[arg-type]


@pytest.fixture
def db() -> FakeDB:
    return FakeDB()


@pytest.fixture
def store(db) -> SalesStore:
    return SalesStore(db, max_bulk=5)


@pytest.fixture
def user_a() -> str:
    return str(ObjectId())


@pytest.fixture
def user_b() -> str:
    return str(ObjectId())


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def mcp_url(store):
    """A real FastMCP HTTP server (JWT-verified) backed by the in-memory store."""
    mcp = create_server(McpSettings(mcp_jwt_secret=MCP_SECRET), store=store)  # type: ignore[arg-type]
    port = _free_port()
    app = mcp.http_app(path="/mcp")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    assert server.started, "MCP test server failed to start"
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    await task
