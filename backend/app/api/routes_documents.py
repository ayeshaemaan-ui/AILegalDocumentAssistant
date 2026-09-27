"""Document endpoints (Tier 1 M1: Document Processing).

Upload, index, browse and delete the corpus. Ingestion runs in a background
task because indexing a large PDF takes tens of seconds - far past a sensible
HTTP timeout - so the upload call returns immediately with a job id and the UI
polls for progress.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.api.schemas import DocumentSummary, IngestRequest
from app.config import get_settings
from app.ingestion.pipeline import get_pipeline
from app.ingestion.taxonomy import category_labels
from app.retrieval.retriever import get_retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".txt", ".docx", ".md"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# In-memory job board. Fine for a single container; a multi-replica deployment
# would move this to Redis or the database.
_JOBS: dict[str, dict[str, Any]] = {}


@router.get("", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    return [DocumentSummary(**doc) for doc in get_pipeline().manifest.list()]


@router.get("/stats")
async def stats() -> dict[str, Any]:
    pipeline = get_pipeline()
    return {**pipeline.stats(), "bm25_chunks": get_retriever().bm25.size}


@router.get("/clause-types")
async def clause_types() -> dict[str, Any]:
    """The 41 CUAD categories, for the UI's filter panel."""
    return {"clause_types": category_labels()}


@router.get("/{doc_id}")
async def get_document(doc_id: str) -> dict[str, Any]:
    doc = get_pipeline().manifest.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No document with id '{doc_id}'.")
    return doc


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict[str, Any]:
    pipeline = get_pipeline()
    if pipeline.manifest.get(doc_id) is None:
        raise HTTPException(status_code=404, detail=f"No document with id '{doc_id}'.")
    removed = pipeline.delete_document(doc_id)
    # BM25 is an in-memory snapshot of the store, so it must be rebuilt or it
    # would keep serving chunks from a document the user just deleted.
    await asyncio.to_thread(get_retriever().refresh_bm25)
    return {"doc_id": doc_id, "chunks_removed": removed}


@router.post("/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload").name
    destination = settings.upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"

    size = 0
    try:
        with destination.open("wb") as handle:
            # Streamed in chunks so a large upload never sits in memory, and the
            # size cap is enforced before the whole file is on disk.
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
                    )
                handle.write(chunk)
    finally:
        await file.close()

    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "job_id": job_id, "status": "queued", "filename": safe_name,
        "started_at": time.time(), "processed": 0, "total": 1, "results": [],
    }
    background.add_task(_run_ingest_job, job_id, [destination])
    return {"job_id": job_id, "filename": safe_name, "size_bytes": size, "status": "queued"}


@router.post("/ingest")
async def ingest_directory(payload: IngestRequest, background: BackgroundTasks) -> dict[str, Any]:
    """Bulk-index a server-side directory - how the CUAD corpus gets loaded."""
    settings = get_settings()
    directory = Path(payload.directory) if payload.directory else settings.cuad_dir / "full_contract_txt"
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    files: list[Path] = []
    for pattern in ("*.txt", "*.pdf", "*.docx"):
        files.extend(sorted(directory.rglob(pattern)))
    files = sorted(set(files))[: payload.limit]
    if not files:
        raise HTTPException(status_code=404, detail=f"No supported documents in {directory}")

    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "job_id": job_id, "status": "queued", "directory": str(directory),
        "started_at": time.time(), "processed": 0, "total": len(files), "results": [],
    }
    background.add_task(_run_ingest_job, job_id, files, payload.strategy, payload.force)
    return {"job_id": job_id, "total": len(files), "status": "queued"}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job with id '{job_id}'.")
    return job


def _run_ingest_job(
    job_id: str, files: list[Path], strategy: str | None = None, force: bool = False
) -> None:
    """Index files one at a time, updating the job board as it goes.

    Runs in a worker thread (FastAPI's BackgroundTasks), so the embedding work
    never blocks the event loop serving chat requests.
    """
    job = _JOBS[job_id]
    job["status"] = "running"
    pipeline = get_pipeline()

    for path in files:
        try:
            result = pipeline.ingest_file(path, strategy=strategy, force=force)
            job["results"].append(result.as_dict())
        except Exception as exc:
            logger.exception("Ingest failed for %s", path)
            job["results"].append({"title": path.name, "error": str(exc), "chunk_count": 0})
        job["processed"] += 1

    try:
        # One rebuild at the end rather than per file - BM25 is a full reindex.
        job["bm25_chunks"] = get_retriever().refresh_bm25()
    except Exception as exc:
        logger.warning("BM25 rebuild failed: %s", exc)
        job["bm25_chunks"] = 0

    indexed = [r for r in job["results"] if r.get("chunk_count")]
    job.update(
        status="completed",
        finished_at=time.time(),
        elapsed_s=round(time.time() - job["started_at"], 1),
        documents_indexed=len(indexed),
        chunks_indexed=sum(r.get("chunk_count", 0) for r in indexed),
        skipped=sum(1 for r in job["results"] if r.get("skipped")),
        failed=sum(1 for r in job["results"] if r.get("error")),
    )
    logger.info(
        "Ingest job %s finished: %d docs, %d chunks in %.1fs",
        job_id, job["documents_indexed"], job["chunks_indexed"], job["elapsed_s"],
    )
