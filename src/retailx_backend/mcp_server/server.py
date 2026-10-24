"""FastMCP server factory and CLI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier
from pymongo import AsyncMongoClient
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from .settings import McpSettings
from .store import SalesStore
from .tools import register_tools

INSTRUCTIONS = (
    "RetailX sales tools. All data is scoped to the authenticated user; never pass user ids. "
    "Amounts are integer paise. Write tools require an approved, operation-bound token."
)


def create_server(settings: McpSettings | None = None, store: SalesStore | None = None) -> FastMCP:
    """Build the MCP server. Pass ``store`` in tests; otherwise a Mongo client is created lazily."""
    settings = settings or McpSettings()  # type: ignore[call-arg]
    auth = JWTVerifier(
        public_key=settings.mcp_jwt_secret.get_secret_value(),
        issuer=settings.mcp_jwt_issuer,
        audience=settings.mcp_jwt_audience,
        algorithm="HS256",
    )

    lifespan = None
    if store is None:
        store = SalesStore(None, max_bulk=settings.max_bulk_affected)

        @asynccontextmanager
        async def lifespan(_server: FastMCP):  # noqa: F811
            client = AsyncMongoClient(settings.mongodb_uri, tz_aware=True)
            store.bind(client[settings.mongodb_db])
            try:
                yield
            finally:
                await client.close()

    mcp = FastMCP("RetailX Sales", instructions=INSTRUCTIONS, auth=auth, lifespan=lifespan)
    register_tools(mcp, store)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    return mcp


def main() -> None:
    settings = McpSettings()  # type: ignore[call-arg]
    create_server(settings).run(transport="http", host=settings.host, port=settings.port, path="/mcp")


if __name__ == "__main__":
    main()
