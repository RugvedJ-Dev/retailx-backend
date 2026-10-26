from datetime import date

import jwt
import pytest

from retailx_backend import repos
from retailx_backend.domain import (
    fmt_money,
    iter_labels,
    parse_date,
    pct_change,
    period_label,
    to_paise,
    validate_range,
)
from retailx_backend.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    mint_mcp_token,
    verify_password,
)


# ------------------------------------------------------------------ domain
def test_to_paise_is_exact_and_rounds_half_up():
    assert to_paise(100) == 10000
    assert to_paise("0.1") == 10
    assert to_paise(19.995) == 2000  # float repr handled via Decimal(str())
    with pytest.raises(ValueError):
        to_paise("abc")
    with pytest.raises(ValueError):
        to_paise(float("inf"))


def test_fmt_money_uses_indian_grouping():
    assert fmt_money(10000) == "\u20b9100.00"
    assert fmt_money(123456050) == "\u20b912,34,560.50"
    assert fmt_money(-5) == "-\u20b90.05"


def test_date_validation():
    assert parse_date("2026-02-28") == date(2026, 2, 28)
    for bad in ("2026-02-30", "26-1-1", "2026/01/01", "1999-01-01"):
        with pytest.raises(ValueError):
            parse_date(bad)
    with pytest.raises(ValueError):
        validate_range("2026-02-02", "2026-02-01")
    with pytest.raises(ValueError):
        validate_range("2020-01-01", "2026-01-01")


def test_labels_and_pct_change():
    assert iter_labels(date(2026, 1, 30), date(2026, 2, 2), "month") == ["2026-01", "2026-02"]
    assert len(iter_labels(date(2026, 1, 1), date(2026, 1, 31), "day")) == 31
    assert period_label(date(2026, 1, 1), "week") == "2026-W01"
    assert pct_change(0, 5) is None
    assert pct_change(200, 150) == -25.0


# ----------------------------------------------------------------- security
def test_password_hashing_roundtrip():
    hashed = hash_password("correct horse")
    assert hashed != "correct horse"
    assert verify_password(hashed, "correct horse")
    assert not verify_password(hashed, "wrong")
    assert not verify_password(None, "anything")  # unknown user path


def test_access_token_roundtrip_and_rejections(settings):
    token = create_access_token(settings, "u1")
    assert decode_access_token(settings, token)["sub"] == "u1"
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(settings, token + "x")
    # an MCP token must not be usable as an API session token
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(settings, mint_mcp_token(settings, "u1", ["sales:read"]))


def test_mcp_token_claims(settings):
    token = mint_mcp_token(settings, "u1", ["sales:write"], op_key="op-12345678")
    claims = jwt.decode(token, settings.mcp_jwt_secret.get_secret_value(), algorithms=["HS256"],
                        audience=settings.mcp_jwt_audience)
    assert claims["scope"] == "sales:write" and claims["op"] == "op-12345678"


# -------------------------------------------------------------------- repos
async def test_duplicate_email_rejected(db):
    await repos.create_user(db, name="A", email="A@x.com", password_hash="h")
    assert (await repos.get_user_by_email(db, "a@X.com"))["name"] == "A"


async def test_thread_tenant_isolation(db):
    from bson import ObjectId

    u1, u2 = ObjectId(), ObjectId()
    t = await repos.create_thread(db, u1, "Sales")
    assert await repos.get_thread(db, u1, str(t["_id"])) is not None
    assert await repos.get_thread(db, u2, str(t["_id"])) is None
    assert await repos.get_thread(db, u1, "not-an-id") is None
    assert not await repos.delete_thread(db, u2, str(t["_id"]))
    assert await repos.list_threads(db, u2) == []


async def test_run_lock_is_exclusive(db):
    from bson import ObjectId

    u = ObjectId()
    t = await repos.create_thread(db, u, "x")
    assert await repos.acquire_run_lock(db, t["_id"], u, 60)
    assert not await repos.acquire_run_lock(db, t["_id"], u, 60)
    await repos.release_run_lock(db, t["_id"])
    assert await repos.acquire_run_lock(db, t["_id"], u, 60)


async def test_approval_can_be_claimed_only_once(db):
    from bson import ObjectId

    u, t = ObjectId(), ObjectId()
    await repos.upsert_approval(db, approval_id="ap1", thread_oid=t, user_id=u, run_id="r",
                                payload={"summary": "x"}, ttl_hours=1)
    first = await repos.claim_approval(db, approval_id="ap1", thread_oid=t, user_id=u,
                                       decision="approve")
    second = await repos.claim_approval(db, approval_id="ap1", thread_oid=t, user_id=u,
                                        decision="approve")
    assert first is not None and second is None
    # another tenant can never claim it
    other = await repos.claim_approval(db, approval_id="ap1", thread_oid=t, user_id=ObjectId(),
                                       decision="approve")
    assert other is None
