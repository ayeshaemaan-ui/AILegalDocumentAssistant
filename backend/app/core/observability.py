"""Observability (Module 8: LangSmith / Helicone / OpenTelemetry).

A RAG answer is the output of eight or nine steps. When one is wrong, "the
answer was bad" is useless - you need to know *which* step failed: did
retrieval miss the clause, did re-ranking bury it, or did generation ignore it?
That is what tracing buys, and it is why every stage in this app opens a span.

Three tools, used for what each is actually good at:

  OpenTelemetry  vendor-neutral spans for the whole request, LLM and non-LLM
                 alike. The only one that traces embedding and retrieval time,
                 which is usually where the latency actually is.
  LangSmith      LLM-native: prompts, completions, and token counts per step,
                 and it doubles as the dataset/experiment store for evaluations.
  Helicone       a reverse proxy, so it needs no code at the call sites at all -
                 it is wired in `core/llm.py` by pointing the SDK's base_url at
                 it. Best for cost dashboards and cache-hit visibility.

All three are optional. With none configured the app still records every span
to a local SQLite table, which is what the /observability page in the UI reads -
so the tracing story works offline with no vendor account.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id    TEXT NOT NULL,
    span_id     TEXT NOT NULL,
    parent_id   TEXT,
    name        TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    duration_ms REAL,
    status      TEXT DEFAULT 'ok',
    session_id  TEXT,
    attributes  TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_trace   ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_session ON spans(session_id);
CREATE INDEX IF NOT EXISTS idx_spans_started ON spans(started_at);
"""


@dataclass(slots=True)
class Span:
    trace_id: str
    span_id: str
    name: str
    started_at: float
    parent_id: str | None = None
    session_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)


class TraceStore:
    """Local span sink - always on, so tracing works with zero vendor setup."""

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

    def write(self, span: Span, ended_at: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO spans
                   (trace_id,span_id,parent_id,name,started_at,ended_at,duration_ms,status,session_id,attributes)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    span.trace_id, span.span_id, span.parent_id, span.name,
                    span.started_at, ended_at, (ended_at - span.started_at) * 1000,
                    span.status, span.session_id,
                    json.dumps(span.attributes, default=str)[:20000],
                ),
            )

    def trace(self, trace_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at", (trace_id,)
            ).fetchall()
        return [_row_to_span(r) for r in rows]

    def recent_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        """Root spans only - one row per request, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT trace_id, session_id, name, started_at, duration_ms, status, attributes
                   FROM spans WHERE parent_id IS NULL
                   ORDER BY started_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [_row_to_span(r) for r in rows]

    def latency_percentiles(self, name: str = "rag.answer", hours: int = 24) -> dict[str, float]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT duration_ms FROM spans WHERE name = ? AND started_at >= ? ORDER BY duration_ms",
                (name, time.time() - hours * 3600),
            ).fetchall()
        values = [float(r["duration_ms"]) for r in rows if r["duration_ms"] is not None]
        if not values:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

        def pct(p: float) -> float:
            idx = min(len(values) - 1, int(round((p / 100) * (len(values) - 1))))
            return round(values[idx], 1)

        return {"count": len(values), "p50": pct(50), "p95": pct(95), "p99": pct(99)}


def _row_to_span(row: sqlite3.Row) -> dict[str, Any]:
    entry = dict(row)
    try:
        entry["attributes"] = json.loads(entry.get("attributes") or "{}")
    except json.JSONDecodeError:
        entry["attributes"] = {}
    return entry


class Tracer:
    """Fan-out tracer: one `span()` call feeds local SQLite and OTel together."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = TraceStore(self.settings.trace_db_path)
        self._otel = self._init_otel()
        self._init_langsmith()

    def _init_otel(self):
        if not self.settings.otel_enabled:
            return None
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(
                resource=Resource.create({"service.name": self.settings.app_name})
            )
            if self.settings.otel_endpoint:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=self.settings.otel_endpoint))
                )
            trace.set_tracer_provider(provider)
            return trace.get_tracer(self.settings.app_name)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.info("OpenTelemetry unavailable (%s); local tracing only.", exc)
            return None

    def _init_langsmith(self) -> None:
        """LangSmith reads env vars, so enabling it is purely configuration.

        Set here rather than in the shell so `LANGSMITH_TRACING=true` in .env is
        enough - LangChain and LangGraph runs are then traced automatically,
        with no decorator on any chain.
        """
        if not (self.settings.langsmith_tracing and self.settings.langsmith_api_key):
            return
        import os

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = self.settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = self.settings.langsmith_project
        logger.info("LangSmith tracing enabled: project=%s", self.settings.langsmith_project)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        parent_id: str | None = None,
        session_id: str | None = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        span = Span(
            trace_id=trace_id or uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent_id,
            name=name,
            started_at=time.time(),
            session_id=session_id,
            attributes=dict(attributes),
        )
        otel_ctx = self._otel.start_as_current_span(name) if self._otel else None
        if otel_ctx:
            otel_ctx.__enter__()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.attributes["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ended = time.time()
            if otel_ctx:
                try:
                    otel_ctx.__exit__(None, None, None)
                except Exception:  # pragma: no cover
                    pass
            try:
                self.store.write(span, ended)
            except Exception as exc:  # tracing must never break the request
                logger.warning("Failed to persist span %s: %s", name, exc)


_tracer: Tracer | None = None


def get_tracer(settings: Settings | None = None) -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer(settings)
    return _tracer


def new_trace_id() -> str:
    return uuid.uuid4().hex
