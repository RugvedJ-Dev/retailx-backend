"""Entrypoint for ``fastmcp run`` / FastMCP Horizon: exposes a module-level ``mcp`` object.

Requires MCP_JWT_SECRET and MONGODB_URI (and optionally MONGODB_DB) in the environment.
"""

from .server import create_server

mcp = create_server()
