"""Session management (Tier 1 M2: Session Management).

The Messages API is stateless - it remembers nothing between calls. Anything
that feels like memory is state the application keeps and replays. This module
is that state.

Sessions are persisted to SQLite rather than held in a dict, for three reasons
that all matter in production: a restart or redeploy does not wipe every open
conversation; two worker processes see the same session; and the conversation
history is on disk where an evaluation run can replay it.

Each session also carries its own retrieval filters, so "only search the
Accuray contract" is a property of the conversation rather than something the
user restates in every message.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    expires_at   REAL NOT NULL,
    title        TEXT,
    summary      TEXT,
    filters_json TEXT,
    meta_json    TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    trace_id    TEXT,
    meta_json   TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry  ON sessions(expires_at);
"""


@dataclass(slots=True)
class Message:
    role: str
    content: str
    created_at: float = field(default_factory=time.time)
    trace_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "trace_id": self.trace_id,
            "meta": self.meta,
        }


@dataclass(slots=True)
class Session:
    session_id: str
    created_at: float
    updated_at: float
    expires_at: float
    title: str | None = None
    summary: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)

    def as_dict(self, include_messages: bool = True) -> dict[str, Any]:
        payload = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "title": self.title,
            "summary": self.summary,
            "filters": self.filters,
            "meta": self.meta,
            "message_count": len(self.messages),
        }
        if include_messages:
            payload["messages"] = [m.as_dict() for m in self.messages]
        return payload


class SessionStore:
    def __init__(self, db_path: Path, ttl_seconds: int) -> None:
        self.db_path = db_path
        self.ttl = ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def create(self, *, title: str | None = None, filters: dict[str, Any] | None = None) -> Session:
        now = time.time()
        session = Session(
            session_id=uuid.uuid4().hex,
            created_at=now,
            updated_at=now,
            expires_at=now + self.ttl,
            title=title,
            filters=filters or {},
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO sessions
                   (session_id,created_at,updated_at,expires_at,title,summary,filters_json,meta_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    session.session_id, now, now, session.expires_at, title, None,
                    json.dumps(session.filters), json.dumps({}),
                ),
            )
        return session

    def get(self, session_id: str, *, include_messages: bool = True) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] < time.time():
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                return None
            session = Session(
                session_id=row["session_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                expires_at=row["expires_at"],
                title=row["title"],
                summary=row["summary"],
                filters=json.loads(row["filters_json"] or "{}"),
                meta=json.loads(row["meta_json"] or "{}"),
            )
            if include_messages:
                messages = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                    (session_id,),
                ).fetchall()
                session.messages = [
                    Message(
                        role=m["role"], content=m["content"], created_at=m["created_at"],
                        trace_id=m["trace_id"], meta=json.loads(m["meta_json"] or "{}"),
                    )
                    for m in messages
                ]
        return session

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        return self.create()

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        trace_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO messages (session_id,role,content,created_at,trace_id,meta_json)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, role, content, now, trace_id, json.dumps(meta or {}, default=str)),
            )
            # Every message extends the TTL - an active conversation should not
            # expire mid-thread just because it started eight hours ago.
            conn.execute(
                "UPDATE sessions SET updated_at = ?, expires_at = ? WHERE session_id = ?",
                (now, now + self.ttl, session_id),
            )

    def set_summary(self, session_id: str, summary: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary, time.time(), session_id),
            )

    def set_title(self, session_id: str, title: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ? AND (title IS NULL OR title = '')",
                (title[:120], session_id),
            )

    def set_filters(self, session_id: str, filters: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET filters_json = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(filters), time.time(), session_id),
            )

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT s.*, COUNT(m.id) AS message_count
                   FROM sessions s LEFT JOIN messages m ON m.session_id = s.session_id
                   WHERE s.expires_at > ?
                   GROUP BY s.session_id ORDER BY s.updated_at DESC LIMIT ?""",
                (time.time(), limit),
            ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "title": r["title"] or "Untitled conversation",
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": r["message_count"],
                "filters": json.loads(r["filters_json"] or "{}"),
            }
            for r in rows
        ]

    def delete(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def purge_expired(self) -> int:
        with self._lock, self._connect() as conn:
            expired = [
                r["session_id"]
                for r in conn.execute(
                    "SELECT session_id FROM sessions WHERE expires_at < ?", (time.time(),)
                ).fetchall()
            ]
            for sid in expired:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
        return len(expired)


_store: SessionStore | None = None


def get_session_store(settings: Settings | None = None) -> SessionStore:
    global _store
    if _store is None:
        settings = settings or get_settings()
        _store = SessionStore(settings.state_db_path, settings.session_ttl_seconds)
    return _store
