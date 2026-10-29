"""Threads (conversations), chat streaming and human-in-the-loop approvals."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .. import repos
from ..agent.runner import approval_view, message_view
from ..schemas import (
    ApprovalDecisionIn,
    ApprovalOut,
    ChatIn,
    MessageOut,
    ThreadCreate,
    ThreadDetail,
    ThreadOut,
)
from .deps import AuthUser, current_user
from .sse import sse_response

router = APIRouter(prefix="/api/threads", tags=["threads"])


def _thread_out(doc: dict) -> ThreadOut:
    return ThreadOut(id=str(doc["_id"]), name=doc["name"], created_at=doc["created_at"].isoformat(),
                     updated_at=doc["updated_at"].isoformat())


async def _owned_thread(request: Request, user: AuthUser, thread_id: str) -> dict:
    thread = await repos.get_thread(request.app.state.db, user.id, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


async def _pending_approval(request: Request, user: AuthUser, thread: dict) -> dict | None:
    """Pending approval record, reconciled against the graph (the source of truth)."""
    st = request.app.state
    doc = await repos.get_pending_approval(st.db, thread["_id"], user.id)
    if doc and doc["status"] == "pending" and await st.runner.pending_interrupt(
            user.id, thread["_id"]) is None:
        await repos.finish_approval(st.db, doc["_id"], "expired")  # orphaned; unblock the thread
        return None
    return doc


@router.get("", response_model=list[ThreadOut])
async def list_threads(request: Request, user: AuthUser = Depends(current_user)) -> list[ThreadOut]:
    return [_thread_out(t) for t in await repos.list_threads(request.app.state.db, user.id)]


@router.post("", response_model=ThreadOut, status_code=201)
async def create_thread(body: ThreadCreate, request: Request,
                        user: AuthUser = Depends(current_user)) -> ThreadOut:
    return _thread_out(await repos.create_thread(request.app.state.db, user.id, body.name))


@router.get("/{thread_id}", response_model=ThreadDetail)
async def get_thread(thread_id: str, request: Request,
                     user: AuthUser = Depends(current_user)) -> ThreadDetail:
    thread = await _owned_thread(request, user, thread_id)
    messages = await repos.list_messages(request.app.state.db, thread["_id"], user.id)
    pending = await _pending_approval(request, user, thread)
    return ThreadDetail(
        thread=_thread_out(thread),
        messages=[MessageOut(**message_view(m)) for m in messages],
        pending_approval=ApprovalOut(**approval_view(pending)) if pending else None,
    )


@router.delete("/{thread_id}", status_code=204)
async def delete_thread(thread_id: str, request: Request,
                        user: AuthUser = Depends(current_user)) -> Response:
    thread = await _owned_thread(request, user, thread_id)
    await repos.delete_thread(request.app.state.db, user.id, thread_id)
    await request.app.state.runner.delete_state(str(thread["_id"]))
    return Response(status_code=204)


@router.post("/{thread_id}/chat")
async def chat(thread_id: str, body: ChatIn, request: Request,
               user: AuthUser = Depends(current_user)) -> StreamingResponse:
    st = request.app.state
    thread = await _owned_thread(request, user, thread_id)
    if await _pending_approval(request, user, thread):
        raise HTTPException(status_code=409, detail="Resolve the pending approval first")
    if not await repos.acquire_run_lock(st.db, thread["_id"], user.id, st.settings.run_lock_seconds):
        raise HTTPException(status_code=409, detail="A response is already being generated")
    try:
        await repos.add_message(st.db, thread_oid=thread["_id"], user_id=user.id, role="user",
                                content=body.message)
        await repos.touch_thread(st.db, thread["_id"])
    except Exception:
        await repos.release_run_lock(st.db, thread["_id"])
        raise
    return sse_response(st.runner.run_chat(user.id, thread["_id"], body.message),
                        lambda: repos.release_run_lock(st.db, thread["_id"]))


@router.post("/{thread_id}/approvals/{approval_id}")
async def decide_approval(thread_id: str, approval_id: str, body: ApprovalDecisionIn,
                          request: Request, user: AuthUser = Depends(current_user)) -> StreamingResponse:
    """Approve or reject a paused operation. Resumes the graph at most once per approval."""
    st = request.app.state
    thread = await _owned_thread(request, user, thread_id)
    approval = await repos.get_approval(st.db, approval_id, user.id)
    if approval is None or approval["thread_id"] != thread["_id"]:
        raise HTTPException(status_code=404, detail="Approval not found")
    if approval["status"] in ("approved", "rejected", "expired"):
        raise HTTPException(status_code=409, detail=f"Approval already {approval['status']}")
    if approval["expires_at"] <= repos.now():
        raise HTTPException(status_code=410, detail="Approval has expired")
    pending = await st.runner.pending_interrupt(user.id, thread["_id"])
    if pending is None or pending.get("approval_id") != approval_id:
        raise HTTPException(status_code=409, detail="No matching pending approval")
    if not await repos.acquire_run_lock(st.db, thread["_id"], user.id, st.settings.run_lock_seconds):
        raise HTTPException(status_code=409, detail="This thread is busy")
    claimed = await repos.claim_approval(st.db, approval_id=approval_id, thread_oid=thread["_id"],
                                         user_id=user.id, decision=body.decision)
    if claimed is None:
        await repos.release_run_lock(st.db, thread["_id"])
        raise HTTPException(status_code=409, detail="Approval is already being processed")
    return sse_response(
        st.runner.run_resume(user.id, thread["_id"], approval_id, body.decision),
        lambda: repos.release_run_lock(st.db, thread["_id"]),
    )
