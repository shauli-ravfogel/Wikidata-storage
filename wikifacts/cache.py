"""A tiny persistent key -> JSON cache on SQLite.

Used to store answer samples of frozen models: a (model, question, sampling params) triple
never needs to be recomputed, so one cache file can be shared by all experiments and
by an RL reward running across many jobs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Optional


def make_key(**parts: Any) -> str:
    """Deterministic key from named parts, e.g. make_key(model=..., question=..., k=8)."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class Cache:
    def __init__(self, path: str):
        # Several jobs may share one file: wait (up to 10 min) for the write lock instead of failing.
        self.conn = sqlite3.connect(path, timeout=600)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str) -> Optional[Any]:
        row = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_many(self, items: dict[str, Any]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO cache VALUES (?,?)",
            [(k, json.dumps(v, ensure_ascii=False)) for k, v in items.items()],
        )
        self.conn.commit()
