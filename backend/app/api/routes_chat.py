"""Chat endpoints (Tier 1 M2: AI APIs Integration + Streaming Responses).

Three ways to ask the same corpus a question, because they have genuinely
different uses:

  POST /api/chat/ask         blocking JSON. Simple clients, tests, evaluation.
  POST /api/chat/stream      Server-Sent Events. What the UI uses.
  POST /api/chat/agent       the multi-step tool-calling agent / orchestrator.

On the choice of SSE over WebSockets: this is one-directional (server pushes
tokens, client says nothing mid-stream), and SSE is plain HTTP - it survives
proxies, needs no connection upgrade, and reconnects on its own. A WebSocket
would be strictly more machinery for strictly less benefit.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import client_key
from app.api.schemas import AgentRequest, AskRequest, AskResponse
from app.core.rate_limit import RateLimitExceeded
from app.core.service import BudgetExceeded, get_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, key: str = Depends(client_key)) -> AskResponse:
    """Run the full RAG graph and return the complete answer."""
    service = get_service()
    try:
        result = await service.answer(
            payload.question,
            session_id=payload.session_id,
            filters=payload.filters(),
            client_key=key,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429, detail=str(exc), headers={"Retry-After": str(int(exc.retry_after) + 1)}
        ) from exc
    except BudgetExceeded as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    return AskResponse(**result)


@router.post("/stream")
async def stream(payload: AskRequest, request: Request, key: str = Depends(client_key)):
    """Stream the answer as Server-Sent Events.

    Frame types the client should handle:
      meta      session_id / trace_id, sent first so the UI can bind the turn
      thinking  summarised reasoning, shown in a collapsible panel
      token     an answer fragment - append to the visible text
      restart   discard buffered text; the answer is being regenerated
      done      the full envelope: citations, retrieval trace, usage, warnings
      error     rate limit, budget, or internal failure
    """
    service = get_service()

    async def event_source():
        try:
            async for event in service.answer_stream(
                payload.question,
                session_id=payload.session_id,
                filters=payload.filters(),
                client_key=key,
            ):
                # A client that navigated away should not keep the graph running.
                if await request.is_disconnected():
                    logger.info("Client disconnected; abandoning stream")
                    break
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'], default=str)}\n\n"
        except Exception:
            logger.exception("Stream failed")
            payload_json = json.dumps(
                {"kind": "internal", "message": "The stream failed unexpectedly."}
            )
            yield f"event: error\ndata: {payload_json}\n\n"
        finally:
            yield "event: close\ndata: {}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx not to buffer, which would defeat streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/agent")
async def run_agent(payload: AgentRequest, key: str = Depends(client_key)) -> dict:
    """Multi-step agent (`agent`) or decompose-and-fan-out (`orchestrator`)."""
    from app.agents.orchestrator import Orchestrator, ResearchAgent
    from app.core.observability import new_trace_id
    from app.memory.session import get_session_store

    service = get_service()
    try:
        service._admit(key)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except BudgetExceeded as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc

    sessions = get_session_store()
    session = sessions.get_or_create(payload.session_id)
    trace_id = new_trace_id()

    runner = (
        Orchestrator()
        if payload.mode == "orchestrator"
        else ResearchAgent(max_steps=payload.max_steps)
    )
    with get_service().tracer.span(
        f"agent.{payload.mode}", trace_id=trace_id, session_id=session.session_id,
        question=payload.question[:300],
    ) as span:
        run = await runner.run(payload.question, session_id=session.session_id, trace_id=trace_id)
        span.set(steps=len(run.steps), cost_usd=run.total_cost_usd, stopped=run.stopped_reason)

    sessions.append_message(session.session_id, "user", payload.question, trace_id=trace_id)
    sessions.append_message(
        session.session_id, "assistant", run.answer, trace_id=trace_id,
        meta={"mode": payload.mode, "steps": len(run.steps)},
    )
    return {"session_id": session.session_id, "trace_id": trace_id, "mode": payload.mode, **run.as_dict()}
