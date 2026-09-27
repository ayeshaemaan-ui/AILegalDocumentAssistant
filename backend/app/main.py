"""FastAPI application entry point.

Startup does the expensive work once - loading the embedding model and building
the BM25 index - so the first user request is not the one that pays for it. The
lifespan handler is also where the background sweep of expired sessions and
stale rate-limit buckets is scheduled.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import (
    routes_admin,
    routes_chat,
    routes_documents,
    routes_eval,
    routes_extract,
)
from app.config import get_settings

logging.basicConfig(
    level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("legallens")


async def _janitor() -> None:
    """Periodic cleanup: expired sessions and idle rate-limit buckets."""
    from app.core.rate_limit import get_rate_limiter
    from app.memory.session import get_session_store

    while True:
        try:
            await asyncio.sleep(900)
            purged = await asyncio.to_thread(get_session_store().purge_expired)
            swept = get_rate_limiter().sweep()
            if purged or swept:
                logger.info("Janitor: %d sessions purged, %d buckets swept", purged, swept)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Janitor pass failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    logger.info("Starting %s (%s)", settings.app_name, settings.environment)

    # Warm the tracer first so startup itself is observable.
    from app.core.observability import get_tracer

    get_tracer(settings)

    try:
        from app.retrieval.retriever import get_retriever

        retriever = get_retriever(settings)
        # Loading BGE takes a few seconds; do it now rather than inside the
        # first user request.
        await asyncio.to_thread(retriever.embeddings.embed_query, "warmup")
        chunks = await asyncio.to_thread(retriever.refresh_bm25)
        logger.info("Retrieval ready: BM25 over %d chunks", chunks)
    except Exception:
        logger.exception("Retrieval warmup failed - the API will still start")

    janitor = asyncio.create_task(_janitor())
    try:
        yield
    finally:
        janitor.cancel()
        logger.info("Shutting down")


settings = get_settings()
app = FastAPI(
    title="LegalLens API",
    description=(
        "Retrieval-Augmented Generation over commercial contracts. "
        "A document research tool - not a substitute for a qualified lawyer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Duration-Ms"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-Duration-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the detail, return a generic message.

    Stack traces and internal paths must not reach the client - they are an
    information-disclosure risk and useless to the user anyway.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. The incident has been logged."},
    )


app.include_router(routes_chat.router)
app.include_router(routes_documents.router)
app.include_router(routes_extract.router)
app.include_router(routes_eval.router)
app.include_router(routes_admin.router)


@app.get("/", tags=["admin"])
async def root() -> dict:
    return {
        "name": settings.app_name,
        "description": "RAG over commercial contracts",
        "docs": "/docs",
        "health": "/api/health",
        "disclaimer": "Document research tool. Not legal advice.",
    }
