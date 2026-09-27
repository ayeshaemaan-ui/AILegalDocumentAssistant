"""Evaluation endpoints (Tier 1 M4: Evaluation Frameworks).

Evaluation is exposed over HTTP, not hidden in a script, because the point is
for it to be run repeatedly - before and after every change to chunking,
retrieval weights or prompts - and compared. The UI's Evaluation page drives
these endpoints directly.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.schemas import EvalRequest
from app.evaluation.runner import get_runner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/eval", tags=["evaluation"])

_JOBS: dict[str, dict[str, Any]] = {}


@router.post("/run")
async def start_run(payload: EvalRequest, background: BackgroundTasks) -> dict[str, Any]:
    """Kick off an evaluation run in the background.

    A full run over 25 examples makes ~100 model calls and takes minutes, which
    is far past an HTTP timeout - so this returns a job id to poll.
    """
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "mode": "retrieval_only" if payload.retrieval_only else "full",
        "processed": 0,
        "total": payload.sample_size,
        "started_at": time.time(),
    }
    background.add_task(_run_eval_job, job_id, payload)
    return _JOBS[job_id]


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No evaluation job '{job_id}'.")
    return job


@router.get("/runs")
async def list_runs(limit: int = 20) -> dict[str, Any]:
    return {"runs": get_runner().list_runs(limit)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run = get_runner().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No evaluation run '{run_id}'.")
    return run


@router.get("/compare")
async def compare(baseline: str, candidate: str) -> dict[str, Any]:
    """Regression check between two runs."""
    try:
        return get_runner().compare_runs(baseline, candidate)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/golden-set")
async def golden_set_stats() -> dict[str, Any]:
    from app.evaluation.golden_set import get_golden_set

    try:
        golden = get_golden_set()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    by_category: dict[str, int] = {}
    for example in golden.examples:
        if not example.is_negative:
            by_category[example.category] = by_category.get(example.category, 0) + 1
    return {
        "total_examples": len(golden),
        "positive_examples": sum(1 for e in golden.examples if not e.is_negative),
        "documents": len({e.doc_id for e in golden.examples}),
        "by_category": dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
    }


async def _run_eval_job(job_id: str, payload: EvalRequest) -> None:
    job = _JOBS[job_id]
    job["status"] = "running"

    def progress(done: int, total: int, _result) -> None:
        job["processed"] = done
        job["total"] = total

    try:
        result = await get_runner().run(
            sample_size=payload.sample_size,
            seed=payload.seed,
            categories=payload.categories or None,
            retrieval_only=payload.retrieval_only,
            judge=payload.judge,
            progress=progress,
        )
        job.update(
            status="completed",
            run_id=result["run_id"],
            metrics=result["metrics"],
            cost_usd=result["cost_usd"],
            elapsed_s=result["elapsed_s"],
            finished_at=time.time(),
        )
    except Exception as exc:
        logger.exception("Evaluation job %s failed", job_id)
        job.update(status="failed", error=str(exc), finished_at=time.time())
