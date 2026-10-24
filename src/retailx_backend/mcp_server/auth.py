"""Caller identity and authorization for MCP tools.

The user id comes ONLY from the signature-verified JWT (``sub``). Tools have no ``user_id``
parameter, so neither the LLM nor the agent can address another tenant's data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token

SCOPE_READ = "sales:read"
SCOPE_WRITE = "sales:write"
_OBJECT_ID = re.compile(r"^[0-9a-f]{24}$")


@dataclass(frozen=True)
class Caller:
    user_id: str
    scopes: frozenset[str]
    op: str | None

    def require(self, scope: str, op_key: str | None = None) -> None:
        if scope not in self.scopes:
            raise ToolError(f"Forbidden: token lacks the '{scope}' scope")
        if scope == SCOPE_WRITE and (op_key is None or self.op != op_key):
            raise ToolError("Forbidden: write token is not bound to this operation")


def caller_from_claims(claims: dict) -> Caller:
    sub = claims.get("sub")
    if not isinstance(sub, str) or not _OBJECT_ID.match(sub):
        raise ToolError("Unauthorized: invalid subject")
    scope = claims.get("scope") or ""
    scopes = frozenset(scope.split()) if isinstance(scope, str) else frozenset(scope)
    op = claims.get("op")
    return Caller(user_id=sub, scopes=scopes, op=op if isinstance(op, str) else None)


def current_caller() -> Caller:
    token = get_access_token()
    if token is None:
        raise ToolError("Unauthorized: authentication required")
    return caller_from_claims(dict(token.claims or {}))
