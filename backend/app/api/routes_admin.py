"""Admin, observability and cost endpoints (Module 8 + Tier 1 M5).

What the UI's Observability and Cost pages read. Everything here is derived
from the spans and usage rows written during normal operation - there is no
separate instrumentation path that could drift from reality.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.core.cost import get_cost_tracker
from app.core.observability import get_tracer
from app.core.rate_limit import get_rate_limiter
from app.memory.session import get_session_store

router = APIRouter(prefix="/api", tags=["admin"])

_STARTED_AT = time.time()


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness + readiness. Used by Docker's healthcheck and CI."""
    settings = get_settings()
    from app.ingestion.pipeline import get_pipeline
    from app.retrieval.retriever import get_retriever

    checks: dict[str, Any] = {}
    try:
        pipeline = get_pipeline()
        stats = pipeline.stats()
        checks["vector_store"] = {"ok": True, **stats}
    except Exception as exc:
        checks["vector_store"] = {"ok": False, "error": str(exc)}

    try:
        checks["bm25"] = {"ok": True, "chunks": get_retriever().bm25.size}
    except Exception as exc:
        checks["bm25"] = {"ok": False, "error": str(exc)}

    # Report the credential for the provider actually in use. Checking only the
    # Anthropic key reported "not configured" while Gemini was happily serving
    # every request - a health check that lies about health is worse than none.
    credential_by_provider = {
        "anthropic": bool(settings.anthropic_api_key),
        "openai": bool(settings.openai_api_key),
        "google": bool(settings.google_api_key),
        "ollama": True,  # local, needs no credential
    }
    checks["llm"] = {
        "provider": settings.llm_provider,
        "answer_model": settings.resolved_answer_model,
        "fast_model": settings.resolved_fast_model,
        # Whether a credential exists - never the credential itself.
        "credential_configured": credential_by_provider.get(settings.llm_provider, False),
    }
    healthy = all(c.get("ok", True) for c in checks.values())
    return {
        "status": "healthy" if healthy else "degraded",
        "uptime_s": round(time.time() - _STARTED_AT, 1),
        "environment": settings.environment,
        "checks": checks,
    }


@router.get("/config")
async def config() -> dict[str, Any]:
    """The active pipeline configuration - shown on the UI's architecture page.

    Deliberately excludes every secret; only the knobs that shape behaviour.
    """
    s = get_settings()
    return {
        "app_name": s.app_name,
        "llm": {
            "provider": s.llm_provider,
            "answer_model": s.resolved_answer_model,
            "fast_model": s.resolved_fast_model,
            "judge_model": s.resolved_judge_model,
            "effort": s.answer_effort,
            "prompt_cache": s.enable_prompt_cache,
        },
        "embeddings": {"model": s.embedding_model, "dim": s.embedding_dim, "backend": s.embedding_backend},
        "chunking": {
            "strategy": s.chunk_strategy, "size": s.chunk_size, "overlap": s.chunk_overlap,
        },
        "retrieval": {
            "vector_backend": s.vector_backend,
            "hybrid": s.hybrid_enabled,
            "dense_weight": s.dense_weight,
            "sparse_weight": s.sparse_weight,
            "rrf_k": s.rrf_k,
            "rerank": s.rerank_enabled,
            "rerank_backend": s.rerank_backend,
            "mmr": s.mmr_enabled,
            "candidate_k": s.retrieval_top_k,
            "final_k": s.final_top_k,
        },
        "limits": {
            "rate_limit_requests": s.rate_limit_requests,
            "rate_limit_window_seconds": s.rate_limit_window_seconds,
            "daily_cost_ceiling_usd": s.daily_cost_ceiling_usd,
        },
        "observability": {
            "langsmith": s.langsmith_tracing,
            "helicone": bool(s.helicone_api_key),
            "otel": s.otel_enabled,
        },
    }


@router.get("/graph")
async def graph_diagram() -> dict[str, Any]:
    """Mermaid source for the LangGraph workflow, rendered in the UI."""
    from app.graph.build import render_mermaid

    return {"mermaid": render_mermaid()}


# ------------------------------------------------------------ cost / usage
@router.get("/cost/summary")
async def cost_summary(days: int = 7, session_id: str | None = None) -> dict[str, Any]:
    tracker = get_cost_tracker()
    settings = get_settings()
    spent = tracker.spent_today()
    return {
        **tracker.summary(days=days, session_id=session_id).as_dict(),
        "spent_today_usd": round(spent, 6),
        "daily_ceiling_usd": settings.daily_cost_ceiling_usd,
        "budget_remaining_usd": round(max(0.0, settings.daily_cost_ceiling_usd - spent), 6),
        "daily_series": tracker.daily_series(14),
    }


# ----------------------------------------------------------- observability
@router.get("/traces")
async def recent_traces(limit: int = 50) -> dict[str, Any]:
    tracer = get_tracer()
    return {
        "traces": tracer.store.recent_traces(limit),
        "latency": tracer.store.latency_percentiles("rag.answer", hours=24),
    }


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str) -> dict[str, Any]:
    spans = get_tracer().store.trace(trace_id)
    if not spans:
        raise HTTPException(status_code=404, detail=f"No trace '{trace_id}'.")
    return {"trace_id": trace_id, "spans": spans}


# --------------------------------------------------------------- sessions
@router.get("/sessions")
async def list_sessions(limit: int = 50) -> dict[str, Any]:
    return {"sessions": get_session_store().list_sessions(limit)}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = get_session_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}' (it may have expired).")
    return session.as_dict()


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    get_session_store().delete(session_id)
    return {"session_id": session_id, "deleted": True}


@router.get("/limits")
async def limits(session_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    limiter = get_rate_limiter()
    key = f"session:{session_id}" if session_id else "ip:unknown"
    return {
        "requests_per_window": settings.rate_limit_requests,
        "window_seconds": settings.rate_limit_window_seconds,
        "burst": settings.rate_limit_burst,
        "remaining": limiter.remaining(key),
    }
