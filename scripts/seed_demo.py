"""Seed demo sales for a registered user so charts have data.

Usage (from the backend repo root, after registering in the app):
    uv run --env-file .env python scripts/seed_demo.py you@example.com --days 120
Re-running is safe: every record has a deterministic idempotency key.
"""

import argparse
import asyncio
import os
import random
from datetime import date, timedelta

from pymongo import AsyncMongoClient

from retailx_backend.mcp_server.store import SalesStore


async def main(email: str, days: int) -> None:
    client = AsyncMongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"), tz_aware=True)
    db = client[os.getenv("MONGODB_DB", "retailx")]
    user = await db.users.find_one({"email": email.lower()})
    if user is None:
        raise SystemExit(f"No user with email {email}. Register in the app first.")
    uid, store, rng, today = str(user["_id"]), SalesStore(db), random.Random(42), date.today()
    inserted = 0
    for offset in range(days):
        day = today - timedelta(days=offset)
        weekend_boost = 1.4 if day.weekday() >= 5 else 1.0
        for i in range(rng.randint(1, 4)):
            paise = int(rng.randint(150, 2500) * weekend_boost) * 100
            result = await store.insert_sale(uid, day.isoformat(), paise, "demo",
                                             f"seed-{uid}-{day.isoformat()}-{i}")
            inserted += 0 if result.replayed else 1
    print(f"Seeded {inserted} new sales for {email} (last {days} days).")
    await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("email")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    asyncio.run(main(args.email, args.days))
