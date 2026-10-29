"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    id: str
    name: str
    email: str


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserOut


class ThreadCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class ThreadOut(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    charts: list[dict[str, Any]] = []
    created_at: str


class ApprovalOut(BaseModel):
    approval_id: str
    thread_id: str
    status: str
    operation: str
    summary: str
    details: dict[str, Any]
    destructive: bool
    warnings: list[str]
    expires_at: str


class ThreadDetail(BaseModel):
    thread: ThreadOut
    messages: list[MessageOut]
    pending_approval: ApprovalOut | None = None


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v


class ApprovalDecisionIn(BaseModel):
    decision: Literal["approve", "reject"]
