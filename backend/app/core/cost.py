"""Cost & usage ledger (Tier 1 M5: Cost Tracking).

Every model call in the app writes one row here. That single discipline is what
makes the rest possible: a per-question cost shown in the UI, a daily spend
ceiling that can actually be enforced, and - the useful one - a breakdown by
*stage*, so you can see that routing and re-ranking are pennies and generation
is the bill.

Cache accounting is tracked separately from plain input tokens because it is
the biggest lever available: a cached system prompt reads at roughly a tenth of
the input price, and the `cache_hit_rate` column is how you prove the caching is
actually working rather than silently invalidating every call.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.core.llm import Usage

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL NOT NULL,
    day               TEXT NOT NULL,
    session_id        TEXT,
    trace_id          TEXT,
    stage             TEXT NOT NULL,
    model             TEXT NOT NULL,
    input_tokens      INTEGER DEFAULT 0,
    output_tokens     INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    cost_usd          REAL DEFAULT 0,
    latency_ms        REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_day     ON usage_events(day);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_events(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_trace   ON usage_events(trace_id);
"""


@dataclass(slots=True)
class CostSummary:
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    calls: int = 0
    by_stage: dict[str, dict[str, Any]] = None
    by_model: dict[str, dict[str, Any]] = None
    cache_hit_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "by_stage": self.by_stage or {},
            "by_model": self.by_model or {},
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }


class CostTracker:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def record(
        self, usage: Usage, *, stage: str, session_id: str | None = None, trace_id: str | None = None
    ) -> float:
        if not usage.model:
            return 0.0
        cost = usage.cost_usd
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO usage_events
                   (ts,day,session_id,trace_id,stage,model,input_tokens,output_tokens,
                    cache_read_tokens,cache_write_tokens,cost_usd,latency_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    time.time(), self._today(), session_id, trace_id, stage, usage.model,
                    usage.input_tokens, usage.output_tokens, usage.cache_read_tokens,
                    usage.cache_write_tokens, cost, usage.latency_ms,
                ),
            )
        return cost

    def spent_today(self) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) AS total FROM usage_events WHERE day = ?",
                (self._today(),),
            ).fetchone()
        return float(row["total"])

    def summary(self, *, days: int = 7, session_id: str | None = None) -> CostSummary:
        where, params = "WHERE 1=1", []
        if session_id:
            where += " AND session_id = ?"
            params.append(session_id)
        else:
            where += " AND ts >= ?"
            params.append(time.time() - days * 86400)

        with self._connect() as conn:
            totals = conn.execute(
                f"""SELECT COUNT(*) AS calls, COALESCE(SUM(cost_usd),0) AS cost,
                           COALESCE(SUM(input_tokens+output_tokens+cache_read_tokens+cache_write_tokens),0) AS tokens,
                           COALESCE(SUM(cache_read_tokens),0) AS cache_read,
                           COALESCE(SUM(input_tokens),0) AS fresh_input
                    FROM usage_events {where}""",
                params,
            ).fetchone()
            stages = conn.execute(
                f"""SELECT stage, COUNT(*) AS calls, COALESCE(SUM(cost_usd),0) AS cost,
                           COALESCE(SUM(input_tokens+output_tokens),0) AS tokens,
                           COALESCE(AVG(latency_ms),0) AS avg_latency
                    FROM usage_events {where} GROUP BY stage ORDER BY cost DESC""",
                params,
            ).fetchall()
            models = conn.execute(
                f"""SELECT model, COUNT(*) AS calls, COALESCE(SUM(cost_usd),0) AS cost
                    FROM usage_events {where} GROUP BY model ORDER BY cost DESC""",
                params,
            ).fetchall()

        cacheable = float(totals["cache_read"]) + float(totals["fresh_input"])
        return CostSummary(
            total_cost_usd=float(totals["cost"]),
            total_tokens=int(totals["tokens"]),
            calls=int(totals["calls"]),
            by_stage={
                r["stage"]: {
                    "calls": r["calls"],
                    "cost_usd": round(float(r["cost"]), 6),
                    "tokens": int(r["tokens"]),
                    "avg_latency_ms": round(float(r["avg_latency"]), 1),
                }
                for r in stages
            },
            by_model={
                r["model"]: {"calls": r["calls"], "cost_usd": round(float(r["cost"]), 6)}
                for r in models
            },
            cache_hit_rate=float(totals["cache_read"]) / cacheable if cacheable else 0.0,
        )

    def daily_series(self, days: int = 14) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT day, COUNT(*) AS calls, COALESCE(SUM(cost_usd),0) AS cost
                   FROM usage_events WHERE ts >= ? GROUP BY day ORDER BY day""",
                (time.time() - days * 86400,),
            ).fetchall()
        return [
            {"day": r["day"], "calls": r["calls"], "cost_usd": round(float(r["cost"]), 6)}
            for r in rows
        ]


_tracker: CostTracker | None = None


def get_cost_tracker(settings: Settings | None = None) -> CostTracker:
    global _tracker
    if _tracker is None:
        settings = settings or get_settings()
        _tracker = CostTracker(settings.state_db_path)
    return _tracker
