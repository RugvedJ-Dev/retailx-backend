"""Minimal async adapter over mongomock so repositories written for pymongo's async API can be
tested without a MongoDB server. Only the methods the app uses are implemented."""

from __future__ import annotations

import mongomock


class _Cursor:
    def __init__(self, cursor):
        self._cursor = cursor

    def sort(self, *args, **kwargs):
        self._cursor = self._cursor.sort(*args, **kwargs)
        return self

    def limit(self, n):
        self._cursor = self._cursor.limit(n)
        return self

    async def to_list(self, length=None):
        return list(self._cursor)


class AsyncCollection:
    def __init__(self, collection):
        self._c = collection

    async def insert_one(self, doc):
        return self._c.insert_one(doc)

    async def find_one(self, *args, **kwargs):
        return self._c.find_one(*args, **kwargs)

    def find(self, *args, **kwargs):
        return _Cursor(self._c.find(*args, **kwargs))

    async def aggregate(self, pipeline):
        return _Cursor(self._c.aggregate(pipeline))

    async def update_one(self, *args, **kwargs):
        return self._c.update_one(*args, **kwargs)

    async def update_many(self, *args, **kwargs):
        return self._c.update_many(*args, **kwargs)

    async def delete_one(self, *args, **kwargs):
        return self._c.delete_one(*args, **kwargs)

    async def delete_many(self, *args, **kwargs):
        return self._c.delete_many(*args, **kwargs)

    async def count_documents(self, *args, **kwargs):
        return self._c.count_documents(*args, **kwargs)

    async def find_one_and_update(self, *args, **kwargs):
        return self._c.find_one_and_update(*args, **kwargs)

    async def create_index(self, keys, **kwargs):
        try:
            return self._c.create_index(keys, **kwargs)
        except NotImplementedError:  # e.g. partialFilterExpression / TTL options
            kwargs.pop("partialFilterExpression", None)
            kwargs.pop("expireAfterSeconds", None)
            return self._c.create_index(keys, **kwargs)


class FakeDB:
    def __init__(self):
        self._db = mongomock.MongoClient(tz_aware=True)["retailx_test"]
        self._cache: dict[str, AsyncCollection] = {}

    def __getattr__(self, name: str) -> AsyncCollection:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._cache:
            self._cache[name] = AsyncCollection(self._db[name])
        return self._cache[name]
