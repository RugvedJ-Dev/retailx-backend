"""FastAPI dependencies: authenticated user resolved from the verified session token."""

from __future__ import annotations

from dataclasses import dataclass

import jwt
from bson import ObjectId
from fastapi import HTTPException, Request

from .. import repos
from ..security import decode_access_token


@dataclass(frozen=True)
class AuthUser:
    id: ObjectId
    name: str
    email: str


def _unauthorized(detail: str = "Not authenticated") -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


async def current_user(request: Request) -> AuthUser:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized()
    try:
        claims = decode_access_token(request.app.state.settings, token)
    except jwt.PyJWTError:
        raise _unauthorized("Invalid or expired token") from None
    user = await repos.get_user(request.app.state.db, claims["sub"])
    if user is None:
        raise _unauthorized("Invalid or expired token")
    return AuthUser(id=user["_id"], name=user["name"], email=user["email"])
