"""Registration, login and session introspection."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import repos
from ..schemas import LoginIn, RegisterIn, TokenOut, UserOut
from ..security import create_access_token, hash_password, verify_password
from .deps import AuthUser, current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _token_response(request: Request, user: dict) -> TokenOut:
    out = UserOut(id=str(user["_id"]), name=user["name"], email=user["email"])
    return TokenOut(access_token=create_access_token(request.app.state.settings, out.id), user=out)


@router.post("/register", response_model=TokenOut, status_code=201)
async def register(body: RegisterIn, request: Request) -> TokenOut:
    password_hash = await asyncio.to_thread(hash_password, body.password)
    try:
        user = await repos.create_user(request.app.state.db, name=body.name, email=body.email,
                                       password_hash=password_hash)
    except repos.EmailTaken:
        raise HTTPException(status_code=409, detail="An account with this email already exists") from None
    return _token_response(request, user)


@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn, request: Request) -> TokenOut:
    limiter = request.app.state.login_limiter
    key = f"{request.client.host if request.client else '-'}:{body.email.lower()}"
    if limiter.blocked(key):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.",
                            headers={"Retry-After": "300"})
    user = await repos.get_user_by_email(request.app.state.db, body.email)
    ok = await asyncio.to_thread(verify_password, user["password_hash"] if user else None, body.password)
    if not (user and ok):
        limiter.record_failure(key)
        raise HTTPException(status_code=401, detail="Invalid email or password")
    limiter.reset(key)
    return _token_response(request, user)


@router.get("/me", response_model=UserOut)
async def me(user: AuthUser = Depends(current_user)) -> UserOut:
    return UserOut(id=str(user.id), name=user.name, email=user.email)
