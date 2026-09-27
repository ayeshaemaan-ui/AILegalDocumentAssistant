"""Clause-matrix extraction endpoints.

A matrix run is many model calls, so it cannot answer inside one HTTP request.
The pattern is the same as ingestion and evaluation: POST starts a background
job and returns an id, the client polls, and the finished result is downloadable
as CSV.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.extraction.matrix import MatrixExtractor, MatrixJob, new_job
from app.ingestion.taxonomy import BY_SLUG

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/extract", tags=["extraction"])

_JOBS: dict[str, MatrixJob] = {}

# A matrix is documents x clauses, so the cell count grows multiplicatively.
# 40 x 12 is already 480 model calls; the cap stops a stray click becoming a
# thousand-call job.
MAX_CELLS = 600


class MatrixRequest(BaseModel):
    doc_ids: list[str] = Field(min_length=1, max_length=60)
    clause_slugs: list[str] = Field(min_length=1, max_length=20)
    concurrency: int = Field(default=4, ge=1, le=12)


@router.post("/matrix")
async def start_matrix(payload: MatrixRequest, background: BackgroundTasks) -> dict:
    unknown = [s for s in payload.clause_slugs if s not in BY_SLUG]
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"Unknown clause types: {', '.join(unknown)}"
        )

    cells = len(payload.doc_ids) * len(payload.clause_slugs)
    if cells > MAX_CELLS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{len(payload.doc_ids)} documents x {len(payload.clause_slugs)} clauses "
                f"= {cells} cells, over the {MAX_CELLS} limit. Narrow the selection."
            ),
        )

    from app.ingestion.pipeline import get_pipeline

    manifest = {d["doc_id"]: d["title"] for d in get_pipeline().manifest.list()}
    missing = [d for d in payload.doc_ids if d not in manifest]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Not indexed: {', '.join(missing[:3])}{'…' if len(missing) > 3 else ''}",
        )

    job = new_job(
        payload.doc_ids,
        payload.clause_slugs,
        {d: manifest[d] for d in payload.doc_ids},
    )
    job.total = cells
    _JOBS[job.job_id] = job
    background.add_task(_run_job, job, payload.concurrency)
    return job.as_dict(include_cells=False)


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, cells: bool = Query(default=True)) -> dict:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No extraction job '{job_id}'.")
    return job.as_dict(include_cells=cells)


@router.get("/jobs/{job_id}/csv", response_class=PlainTextResponse)
async def download_csv(job_id: str) -> PlainTextResponse:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No extraction job '{job_id}'.")
    if job.status not in {"completed", "failed"}:
        raise HTTPException(
            status_code=409,
            detail=f"Job is still {job.status} ({job.processed}/{job.total} cells).",
        )
    return PlainTextResponse(
        job.to_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="clause-matrix-{job_id}.csv"'},
    )


@router.get("/jobs")
async def list_jobs(limit: int = 20) -> dict:
    jobs = sorted(_JOBS.values(), key=lambda j: -j.started_at)[:limit]
    return {"jobs": [j.as_dict(include_cells=False) for j in jobs]}


async def _run_job(job: MatrixJob, concurrency: int) -> None:
    try:
        await MatrixExtractor().run(job, concurrency=concurrency)
    except Exception as exc:
        logger.exception("Matrix job %s crashed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
