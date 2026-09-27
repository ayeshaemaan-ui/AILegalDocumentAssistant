"""RAG service - the seam between HTTP and the graph.

Everything that is about *serving* a request rather than answering it lives
here: rate limiting, the cost circuit breaker, session load/save, memory, and
turning the graph's execution into a token stream.

The streaming design is worth explaining. LangGraph runs nodes to completion,
so it cannot itself emit tokens mid-node. Instead the service creates an
`asyncio.Queue`, passes it to the graph through `config.configurable`, and runs
the graph as a background task. The `generate` node pushes tokens onto the
queue as Claude produces them while the service drains it into SSE frames. The
same graph therefore serves the live UI and the batch evaluation runner - the
only difference is whether a queue was supplied.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from app.config import Settings, get_settings
from app.core.cost import get_cost_tracker
from app.core.observability import get_tracer, new_trace_id
from app.core.rate_limit import RateLimitExceeded, get_cost_breaker, get_rate_limiter
from app.graph.build import get_graph
from app.graph.state import initial_state
from app.memory.chat_memory import ConversationMemory
from app.memory.session import get_session_store

logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    pass


class RAGService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.sessions = get_session_store(self.settings)
        self.memory = ConversationMemory(self.settings)
        self.tracker = get_cost_tracker(self.settings)
        self.tracer = get_tracer(self.settings)

    # -------------------------------------------------------- admission
    def _admit(self, client_key: str) -> None:
        """Rate limit, then budget. Both raise rather than degrade silently."""
        allowed, retry_after = get_rate_limiter(self.settings).consume(client_key)
        if not allowed:
            raise RateLimitExceeded(
                retry_after,
                self.settings.rate_limit_requests,
                self.settings.rate_limit_window_seconds,
            )
        ok, message = get_cost_breaker(self.settings).check(self.tracker.spent_today())
        if not ok:
            raise BudgetExceeded(message or "Daily budget exhausted.")

    async def _prepare(
        self, question: str, session_id: str | None, filters: dict[str, Any] | None
    ):
        session = self.sessions.get_or_create(session_id)
        if not session.messages:
            self.sessions.set_title(session.session_id, self.memory.derive_title(question))
        if filters is not None:
            self.sessions.set_filters(session.session_id, filters)
            session.filters = filters

        await self.memory.maybe_summarize(session)
        state = initial_state(
            question,
            session_id=session.session_id,
            trace_id=new_trace_id(),
            filters=session.filters,
            history=self.memory.as_api_messages(session),
            history_summary=self.memory.context_summary(session),
        )
        return session, state

    def _persist(self, session_id: str, question: str, state: dict[str, Any], trace_id: str) -> None:
        self.sessions.append_message(session_id, "user", question, trace_id=trace_id)
        self.sessions.append_message(
            session_id,
            "assistant",
            state.get("final_answer") or state.get("answer", ""),
            trace_id=trace_id,
            meta={
                "intent": state.get("intent"),
                "citations": len(state.get("citations", [])),
                "groundedness": (state.get("groundedness") or {}).get("verdict"),
                "warnings": state.get("warnings", []),
            },
        )

    # ------------------------------------------------------ non-streaming
    async def answer(
        self,
        question: str,
        *,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        client_key: str = "anonymous",
        persist: bool = True,
    ) -> dict[str, Any]:
        """Run the graph to completion. Used by the API's /ask and by evaluation."""
        self._admit(client_key)
        session, state = await self._prepare(question, session_id, filters)
        trace_id = state["trace_id"]

        with self.tracer.span(
            "rag.answer", trace_id=trace_id, session_id=session.session_id, question=question[:300]
        ) as span:
            started = time.perf_counter()
            result = await get_graph().ainvoke(
                state, config={"configurable": {"thread_id": session.session_id}}
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            span.set(
                intent=result.get("intent"),
                citations=len(result.get("citations", [])),
                groundedness=(result.get("groundedness") or {}).get("verdict"),
                elapsed_ms=elapsed_ms,
            )

        if persist:
            self._persist(session.session_id, question, result, trace_id)
        return self._envelope(result, session.session_id, trace_id, elapsed_ms)

    # ---------------------------------------------------------- streaming
    async def answer_stream(
        self,
        question: str,
        *,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        client_key: str = "anonymous",
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-shaped events: meta, stage, token, thinking, done, error."""
        try:
            self._admit(client_key)
        except RateLimitExceeded as exc:
            yield {"event": "error", "data": {"kind": "rate_limit", "message": str(exc), "retry_after": exc.retry_after}}
            return
        except BudgetExceeded as exc:
            yield {"event": "error", "data": {"kind": "budget", "message": str(exc)}}
            return

        session, state = await self._prepare(question, session_id, filters)
        trace_id = state["trace_id"]
        yield {"event": "meta", "data": {"session_id": session.session_id, "trace_id": trace_id}}

        queue: asyncio.Queue = asyncio.Queue()
        result_holder: dict[str, Any] = {}

        async def run_graph() -> None:
            try:
                with self.tracer.span(
                    "rag.answer", trace_id=trace_id, session_id=session.session_id,
                    question=question[:300], streaming=True,
                ) as span:
                    started = time.perf_counter()
                    result = await get_graph().ainvoke(
                        state,
                        config={
                            "configurable": {
                                "thread_id": session.session_id,
                                "stream_queue": queue,
                            }
                        },
                    )
                    result_holder["result"] = result
                    result_holder["elapsed_ms"] = (time.perf_counter() - started) * 1000
                    span.set(
                        intent=result.get("intent"),
                        citations=len(result.get("citations", [])),
                        groundedness=(result.get("groundedness") or {}).get("verdict"),
                    )
            except Exception as exc:
                logger.exception("Graph execution failed")
                result_holder["error"] = exc
            finally:
                # Sentinel: tells the drain loop the producer is finished.
                await queue.put(None)

        task = asyncio.create_task(run_graph())

        # Retrying generation after a failed groundedness check re-enters the
        # generate node, so tokens would arrive twice. Buffer per attempt and
        # tell the client to reset rather than appending.
        emitted_any = False
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if item["type"] == "token":
                    emitted_any = True
                    yield {"event": "token", "data": {"text": item["text"]}}
                elif item["type"] == "thinking":
                    yield {"event": "thinking", "data": {"text": item["text"]}}
                elif item["type"] == "restart":
                    yield {"event": "restart", "data": {"reason": item.get("reason", "regenerating")}}
        finally:
            await task

        if "error" in result_holder:
            yield {
                "event": "error",
                "data": {"kind": "internal", "message": "The request failed while generating an answer."},
            }
            return

        result = result_holder.get("result", {})
        self._persist(session.session_id, question, result, trace_id)
        envelope = self._envelope(
            result, session.session_id, trace_id, result_holder.get("elapsed_ms", 0.0)
        )

        # If nothing streamed (a refusal, or a non-generating route), the client
        # has no text yet - the done frame carries it.
        envelope["streamed"] = emitted_any
        yield {"event": "done", "data": envelope}

    # ----------------------------------------------------------- envelope
    def _envelope(
        self, result: dict[str, Any], session_id: str, trace_id: str, elapsed_ms: float
    ) -> dict[str, Any]:
        usage = result.get("usage", [])
        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "answer": result.get("final_answer") or result.get("answer", ""),
            "citations": result.get("citations", []),
            "intent": result.get("intent"),
            "route_reason": result.get("route_reason"),
            "search_query": result.get("search_query"),
            "clause_types": result.get("clause_types", []),
            "retrieval": result.get("retrieval_trace", {}),
            "groundedness": result.get("groundedness", {}),
            "guardrail": result.get("guardrail", {}),
            "warnings": result.get("warnings", []),
            "blocked": bool(result.get("blocked")),
            "elapsed_ms": round(elapsed_ms, 1),
            "usage": {
                "calls": len(usage),
                "total_tokens": sum(u.get("total_tokens", 0) for u in usage),
                "cost_usd": round(sum(u.get("cost_usd", 0.0) for u in usage), 6),
                "by_stage": usage,
            },
            "events": result.get("events", []),
        }


_service: RAGService | None = None


def get_service(settings: Settings | None = None) -> RAGService:
    global _service
    if _service is None:
        _service = RAGService(settings)
    return _service
