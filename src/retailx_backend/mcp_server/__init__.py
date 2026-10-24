"""RetailX MCP server: validated, user-scoped sales tools backed by MongoDB.

This package is deployable on its own (``retailx-mcp`` / ``fastmcp run``) and never imports the
agent or API layers. It trusts only the verified JWT presented by the caller.
"""
