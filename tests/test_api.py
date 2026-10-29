import json
from bson import ObjectId
from types import SimpleNamespace

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from retailx_backend import repos
from retailx_backend.api.app import create_app
from tests.test_agent import P, S, ScriptedLLM

PASSWORD = "password123"


def parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        data = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
        if data:
            events.append(json.loads("".join(data)))
    return events


@pytest.fixture
async def api(settings, db, mcp_url, store):
    llm = ScriptedLLM([])
    app = create_app(settings.model_copy(update={"mcp_url": mcp_url, "mcp_timeout_seconds": 5}),
                     db=db, checkpointer=InMemorySaver(), llm=llm)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            yield SimpleNamespace(http=client, llm=llm, db=db, store=store)


async def signup(api, email="a@example.com", name="Asha"):
    r = await api.http.post("/api/auth/register",
                            json={"name": name, "email": email, "password": PASSWORD})
    assert r.status_code == 201, r.text
    body = r.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


async def new_thread(api, headers, name="Sales"):
    r = await api.http.post("/api/threads", json={"name": name}, headers=headers)
    assert r.status_code == 201
    return r.json()["id"]


async def chat(api, headers, tid, text="hello"):
    return await api.http.post(f"/api/threads/{tid}/chat", json={"message": text}, headers=headers)


# ----------------------------------------------------------------------------- auth
async def test_register_login_me_and_password_is_hashed(api):
    headers, user = await signup(api)
    me = await api.http.get("/api/auth/me", headers=headers)
    assert me.status_code == 200 and me.json() == user
    login = await api.http.post("/api/auth/login", json={"email": "A@Example.com", "password": PASSWORD})
    assert login.status_code == 200 and login.json()["user"]["id"] == user["id"]
    stored = await api.db.users.find_one({"email": "a@example.com"})
    assert stored["password_hash"].startswith("$argon2") and PASSWORD not in str(stored)


async def test_auth_validation_and_errors(api):
    await signup(api)
    dup = await api.http.post("/api/auth/register", json={"name": "x", "email": "a@example.com",
                                                           "password": PASSWORD})
    assert dup.status_code == 409
    assert (await api.http.post("/api/auth/register", json={
        "name": "x", "email": "b@example.com", "password": "short"})).status_code == 422
    assert (await api.http.post("/api/auth/register", json={
        "name": "x", "email": "not-an-email", "password": PASSWORD})).status_code == 422
    wrong = await api.http.post("/api/auth/login", json={"email": "a@example.com", "password": "nope-nope"})
    unknown = await api.http.post("/api/auth/login", json={"email": "zz@example.com", "password": "nope-nope"})
    assert wrong.status_code == unknown.status_code == 401 and wrong.json() == unknown.json()


async def test_protected_routes_need_a_valid_token(api):
    assert (await api.http.get("/api/threads")).status_code == 401
    bad = {"Authorization": "Bearer not.a.token"}
    assert (await api.http.get("/api/threads", headers=bad)).status_code == 401
    headers, user = await signup(api)
    await api.db.users.delete_one({"_id": ObjectId(user["id"])})  # token for a deleted account
    assert (await api.http.get("/api/threads", headers=headers)).status_code == 401


async def test_login_is_rate_limited_after_repeated_failures(api):
    await signup(api)
    for _ in range(5):
        r = await api.http.post("/api/auth/login", json={"email": "a@example.com", "password": "wrong-pass"})
        assert r.status_code == 401
    locked = await api.http.post("/api/auth/login", json={"email": "a@example.com", "password": PASSWORD})
    assert locked.status_code == 429


async def test_health_and_request_id(api):
    r = await api.http.get("/api/health", headers={"x-request-id": "abc123"})
    assert r.json() == {"status": "ok"} and r.headers["x-request-id"] == "abc123"


# --------------------------------------------------------------------------- threads
async def test_thread_crud_and_tenant_isolation(api):
    ha, _ = await signup(api, "a@example.com")
    hb, _ = await signup(api, "b@example.com", "Bala")
    tid = await new_thread(api, ha)
    assert [t["id"] for t in (await api.http.get("/api/threads", headers=ha)).json()] == [tid]
    assert (await api.http.get("/api/threads", headers=hb)).json() == []
    assert (await api.http.get(f"/api/threads/{tid}", headers=hb)).status_code == 404
    assert (await api.http.delete(f"/api/threads/{tid}", headers=hb)).status_code == 404
    assert (await chat(api, hb, tid)).status_code == 404
    approve = await api.http.post(f"/api/threads/{tid}/approvals/x", json={"decision": "approve"}, headers=hb)
    assert approve.status_code == 404
    assert (await api.http.get("/api/threads/not-an-id", headers=ha)).status_code == 404
    assert (await api.http.post("/api/threads", json={"name": "  "}, headers=ha)).status_code == 422
    assert (await api.http.delete(f"/api/threads/{tid}", headers=ha)).status_code == 204
    assert (await api.http.get(f"/api/threads/{tid}", headers=ha)).status_code == 404


async def test_chat_streams_sse_events_and_history_is_persisted(api):
    headers, _ = await signup(api)
    tid = await new_thread(api, headers)
    api.llm.plans = [P(reply="Hello! Ask me about your sales.")]
    r = await chat(api, headers, tid, "hi there")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e["type"] for e in events] == ["run_started", "token", "message_complete", "done"]
    detail = (await api.http.get(f"/api/threads/{tid}", headers=headers)).json()
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "hi there"), ("assistant", "Hello! Ask me about your sales.")]
    assert detail["pending_approval"] is None


async def test_chat_input_validation_and_run_lock(api):
    headers, user = await signup(api)
    tid = await new_thread(api, headers)
    assert (await chat(api, headers, tid, "   ")).status_code == 422
    assert (await chat(api, headers, tid, "x" * 2001)).status_code == 422
    await repos.acquire_run_lock(api.db, ObjectId(tid), ObjectId(user["id"]), 60)
    assert (await chat(api, headers, tid)).status_code == 409  # a run is already active


async def test_chart_events_are_streamed(api):
    headers, user = await signup(api)
    await api.store.insert_sale(user["id"], "2026-01-02", 5000, "", "seed-op-00000001")
    tid = await new_thread(api, headers)
    api.llm.plans = [P(S(1, "sales_trend", start_date="2026-01-01", end_date="2026-01-03"),
                       S(2, "build_chart", source_steps=["s1"], chart_title="Jan"))]
    events = parse_sse((await chat(api, headers, tid, "visualize")).text)
    chart = next(e for e in events if e["type"] == "chart")["chart"]
    assert chart["type"] == "line" and chart["data"][1] == {"label": "2026-01-02", "total": 50.0, "count": 1}
    final = (await api.http.get(f"/api/threads/{tid}", headers=headers)).json()["messages"][-1]
    assert final["charts"][0]["title"] == "Jan"  # charts persist with the message


# ------------------------------------------------------------------------------ HITL
async def request_insert(api, headers, tid):
    api.llm.plans = [P(S(1, "insert_sale", date="2026-10-29", amount=100))]
    events = parse_sse((await chat(api, headers, tid, "Add Rs 100 to today's sales")).text)
    assert [e["type"] for e in events][-2:] == ["approval_required", "done"]
    return events[-2]["approval"]


async def test_full_approval_flow_over_http(api):
    headers, user = await signup(api)
    tid = await new_thread(api, headers)
    approval = await request_insert(api, headers, tid)
    url = f"/api/threads/{tid}/approvals/{approval['approval_id']}"
    assert (await api.store.sales_for_date(user["id"], "2026-10-29")).total_count == 0

    detail = (await api.http.get(f"/api/threads/{tid}", headers=headers)).json()
    assert detail["pending_approval"]["approval_id"] == approval["approval_id"]  # survives reload
    assert (await chat(api, headers, tid)).status_code == 409  # must resolve first
    assert (await api.http.post(url, json={"decision": "maybe"}, headers=headers)).status_code == 422

    ok = await api.http.post(url, json={"decision": "approve"}, headers=headers)
    events = parse_sse(ok.text)
    assert ok.status_code == 200 and "message_complete" in [e["type"] for e in events]
    assert (await api.store.sales_for_date(user["id"], "2026-10-29")).total_count == 1

    again = await api.http.post(url, json={"decision": "approve"}, headers=headers)
    assert again.status_code == 409  # retried approval does not execute twice
    assert (await api.store.sales_for_date(user["id"], "2026-10-29")).total_count == 1
    assert (await api.http.get(f"/api/threads/{tid}", headers=headers)).json()["pending_approval"] is None


async def test_rejection_over_http_executes_nothing(api):
    headers, user = await signup(api)
    tid = await new_thread(api, headers)
    approval = await request_insert(api, headers, tid)
    r = await api.http.post(f"/api/threads/{tid}/approvals/{approval['approval_id']}",
                            json={"decision": "reject"}, headers=headers)
    assert "cancelled" in parse_sse(r.text)[-2]["message"]["content"]
    assert (await api.store.sales_for_date(user["id"], "2026-10-29")).total_count == 0
    api.llm.plans = [P(reply="ok")]
    assert (await chat(api, headers, tid)).status_code == 200  # thread usable again


async def test_another_user_cannot_resolve_my_approval(api):
    ha, ua = await signup(api, "a@example.com")
    hb, _ = await signup(api, "b@example.com", "Bala")
    tid = await new_thread(api, ha)
    approval = await request_insert(api, ha, tid)
    url = f"/api/threads/{tid}/approvals/{approval['approval_id']}"
    assert (await api.http.post(url, json={"decision": "approve"}, headers=hb)).status_code == 404
    # ...even with a thread of their own
    other = await new_thread(api, hb)
    stolen = f"/api/threads/{other}/approvals/{approval['approval_id']}"
    assert (await api.http.post(stolen, json={"decision": "approve"}, headers=hb)).status_code == 404
    assert (await api.store.sales_for_date(ua["id"], "2026-10-29")).total_count == 0


async def test_expired_approval_cannot_be_used(api):
    from datetime import timedelta

    headers, user = await signup(api)
    tid = await new_thread(api, headers)
    approval = await request_insert(api, headers, tid)
    await api.db.approvals.update_one({"_id": approval["approval_id"]},
                                      {"$set": {"expires_at": repos.now() - timedelta(hours=1)}})
    r = await api.http.post(f"/api/threads/{tid}/approvals/{approval['approval_id']}",
                            json={"decision": "approve"}, headers=headers)
    assert r.status_code == 410
    assert (await api.store.sales_for_date(user["id"], "2026-10-29")).total_count == 0


async def test_deleting_a_thread_removes_its_data(api):
    headers, _ = await signup(api)
    tid = await new_thread(api, headers)
    await request_insert(api, headers, tid)
    assert (await api.http.delete(f"/api/threads/{tid}", headers=headers)).status_code == 204
    assert await api.db.messages.find_one({"thread_id": ObjectId(tid)}) is None
    assert await api.db.approvals.find_one({"thread_id": ObjectId(tid)}) is None
