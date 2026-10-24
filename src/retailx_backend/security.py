"""Password hashing and JWT helpers (API sessions and short-lived MCP tokens)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from .config import Settings

_hasher = PasswordHasher()
# Verified against when the e-mail is unknown so response time doesn't reveal account existence.
_DUMMY_HASH = _hasher.hash("retailx-dummy-password")

SCOPE_READ = "sales:read"
SCOPE_WRITE = "sales:write"


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerificationError, InvalidHashError):
        return False


def create_access_token(settings: Settings, user_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iss": settings.jwt_issuer,
        "typ": "access",
        "iat": now,
        "exp": now + settings.access_token_ttl_minutes * 60,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm="HS256")


def decode_access_token(settings: Settings, token: str) -> dict:
    """Validate signature, expiry, issuer and token type. Raises ``jwt.PyJWTError``."""
    claims = jwt.decode(
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=["HS256"],
        issuer=settings.jwt_issuer,
        options={"require": ["exp", "sub", "iss"]},
    )
    if claims.get("typ") != "access":
        raise jwt.InvalidTokenError("Not an access token")
    return claims


def mint_mcp_token(
    settings: Settings,
    user_id: str,
    scopes: Iterable[str],
    op_key: str | None = None,
) -> str:
    """Mint a short-lived token the MCP server trusts for exactly this user.

    Read tokens carry only ``sales:read``. A ``sales:write`` token is minted only after a human
    approved the operation and is bound to that operation's idempotency key (``op`` claim).
    """
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iss": settings.mcp_jwt_issuer,
        "aud": settings.mcp_jwt_audience,
        "scope": " ".join(scopes),
        "iat": now,
        "exp": now + settings.mcp_token_ttl_seconds,
        "jti": uuid.uuid4().hex,
    }
    if op_key:
        payload["op"] = op_key
    return jwt.encode(payload, settings.mcp_jwt_secret.get_secret_value(), algorithm="HS256")
