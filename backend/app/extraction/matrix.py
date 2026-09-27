"""Clause matrix extraction — contracts × clauses → a reviewable grid → CSV.

This is the task CUAD itself is built around: `master_clauses.csv` is exactly
this shape, 510 contracts down the side and 41 clause categories across the top.
Reviewing contracts one question at a time is what the chat interface is for;
this is for the other half of the job, where you need the same twelve answers
out of forty agreements and then want them in a spreadsheet.

The whole thing is assembled from parts that already exist, which is the point
of having built them properly:

  metadata filtering   scopes retrieval to one document AND one clause type, so
                       each cell searches a handful of chunks instead of 4,200
  the hybrid retriever unchanged - same dense + sparse + re-rank path as chat
  structured outputs   every cell is a validated object, never parsed prose
  the cost ledger      each cell's usage is recorded under its own stage

Concurrency is bounded rather than unlimited: a 40 × 12 matrix is 480 model
calls, and firing those at once would trip provider rate limits and make the
progress bar meaningless.
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from app.config import Settings, get_settings
from app.core.cost import get_cost_tracker
from app.core.llm import Usage, get_llm_client
from app.extraction.schemas import ClauseExtraction
from app.ingestion.taxonomy import BY_SLUG
from app.prompts.extraction import EXTRACTION_SYSTEM, build_extraction_turn
from app.retrieval.retriever import RetrievalFilters, get_retriever

logger = logging.getLogger(__name__)

# Chunks fed to one cell. Small on purpose: the retrieval is already filtered to
# one document and one clause type, so the third result is usually noise, and a
# tighter prompt measurably reduces invented values.
CHUNKS_PER_CELL = 3


@dataclass(slots=True)
class Cell:
    doc_id: str
    clause_slug: str
    status: str = "unsure"
    value: str = ""
    quote: str = ""
    confidence: str = "medium"
    section_path: str | None = None
    page_number: int | None = None
    page_confidence: str = "estimated"
    chunk_id: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "clause_slug": self.clause_slug,
            "status": self.status,
            "value": self.value,
            "quote": self.quote,
            "confidence": self.confidence,
            "section_path": self.section_path,
            "page_number": self.page_number,
            "page_confidence": self.page_confidence,
            "chunk_id": self.chunk_id,
            "error": self.error,
        }

    @property
    def display(self) -> str:
        """What the cell reads as in the table and the CSV.

        The three statuses stay distinguishable all the way to the spreadsheet.
        Collapsing "not present" and "unsure" into a blank would destroy the
        one thing a reviewer most needs to see.
        """
        if self.error:
            return "error"
        if self.status == "found":
            return self.value or "found"
        if self.status == "not_present":
            return "—"
        return "?"


@dataclass(slots=True)
class MatrixJob:
    job_id: str
    doc_ids: list[str]
    clause_slugs: list[str]
    status: str = "queued"
    processed: int = 0
    total: int = 0
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    titles: dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cost_usd: float = 0.0
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        counts = {"found": 0, "not_present": 0, "unsure": 0, "error": 0}
        for cell in self.cells.values():
            counts["error" if cell.error else cell.status] += 1
        return counts

    def as_dict(self, include_cells: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "processed": self.processed,
            "total": self.total,
            "doc_ids": self.doc_ids,
            "clause_slugs": self.clause_slugs,
            "titles": self.titles,
            "clause_labels": {
                slug: (BY_SLUG[slug].label if slug in BY_SLUG else slug)
                for slug in self.clause_slugs
            },
            "summary": self.summary(),
            "cost_usd": round(self.cost_usd, 6),
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
            "error": self.error,
        }
        if include_cells:
            payload["cells"] = [c.as_dict() for c in self.cells.values()]
        return payload

    def to_csv(self) -> str:
        """One row per contract, one column per clause, plus a citation column.

        The citation column is why this is worth more than a plain value grid:
        a reviewer opening the spreadsheet can jump straight to the section the
        value came from instead of re-reading the contract.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        header = ["Document"]
        for slug in self.clause_slugs:
            label = BY_SLUG[slug].label if slug in BY_SLUG else slug
            header += [label, f"{label} — source"]
        writer.writerow(header)

        for doc_id in self.doc_ids:
            row: list[str] = [self.titles.get(doc_id, doc_id)]
            for slug in self.clause_slugs:
                cell = self.cells.get((doc_id, slug))
                if cell is None:
                    row += ["", ""]
                    continue
                row.append(cell.display)
                row.append(_source_label(cell))
            writer.writerow(row)

        return buffer.getvalue()


def _source_label(cell: Cell) -> str:
    if cell.status != "found":
        return ""
    parts: list[str] = []
    if cell.section_path:
        parts.append(f"§ {cell.section_path}")
    if cell.page_number is not None:
        prefix = "p." if cell.page_confidence == "exact" else "~p."
        parts.append(f"{prefix}{cell.page_number}")
    return " · ".join(parts)


class MatrixExtractor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def extract_cell(self, doc_id: str, clause_slug: str, title: str) -> Cell:
        cell = Cell(doc_id=doc_id, clause_slug=clause_slug)
        entry = BY_SLUG.get(clause_slug)
        if entry is None:
            cell.error = f"unknown clause type '{clause_slug}'"
            return cell

        # Both filters at once: this document, this clause type. That is what
        # turns a fuzzy search over the corpus into a precise one over a few
        # chunks, and it is only possible because ingestion tagged them.
        try:
            chunks, _ = await get_retriever(self.settings).retrieve(
                f"{entry.label}. {entry.description}",
                filters=RetrievalFilters(doc_ids=[doc_id], clause_types=[clause_slug]),
                top_k=CHUNKS_PER_CELL,
            )
        except Exception as exc:
            logger.exception("Retrieval failed for %s/%s", doc_id, clause_slug)
            cell.error = f"retrieval: {exc}"
            return cell

        if not chunks:
            # No tagged chunk for this clause in this contract. Reported as
            # "unsure", not "not present": the tagger has moderate recall, so
            # absence of a tag is not evidence of absence of a clause.
            cell.status = "unsure"
            cell.confidence = "low"
            return cell

        try:
            extraction, usage = await get_llm_client(self.settings).structured(
                system=EXTRACTION_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": build_extraction_turn(
                            clause_label=entry.label,
                            clause_description=entry.description,
                            document_title=title,
                            chunks=chunks,
                        ),
                    }
                ],
                schema=ClauseExtraction,
                model=self.settings.resolved_fast_model,
                max_tokens=self.settings.fast_max_tokens,
            )
        except Exception as exc:
            logger.exception("Extraction failed for %s/%s", doc_id, clause_slug)
            cell.error = str(exc)[:200]
            return cell

        get_cost_tracker().record(usage, stage="extract.cell")

        cell.status = extraction.status
        cell.value = extraction.value.strip()
        cell.quote = extraction.quote.strip()
        cell.confidence = extraction.confidence

        # Attach the citation from the excerpt the model said it used.
        index = extraction.excerpt_index
        if extraction.status == "found" and 1 <= index <= len(chunks):
            source = chunks[index - 1]
            cell.chunk_id = source.chunk_id
            cell.section_path = source.metadata.get("section_path") or source.metadata.get("section_number")
            page = source.metadata.get("page_number")
            cell.page_number = int(page) if page is not None else None
            cell.page_confidence = source.metadata.get("page_confidence", "estimated")
        return cell

    async def run(
        self,
        job: MatrixJob,
        *,
        concurrency: int = 4,
        progress: Callable[[MatrixJob], None] | None = None,
    ) -> MatrixJob:
        job.status = "running"
        job.total = len(job.doc_ids) * len(job.clause_slugs)
        semaphore = asyncio.Semaphore(concurrency)
        before = get_cost_tracker().spent_today()

        async def one(doc_id: str, slug: str) -> None:
            async with semaphore:
                cell = await self.extract_cell(doc_id, slug, job.titles.get(doc_id, doc_id))
            job.cells[(doc_id, slug)] = cell
            job.processed += 1
            if progress:
                progress(job)

        try:
            await asyncio.gather(
                *(one(d, s) for d in job.doc_ids for s in job.clause_slugs)
            )
            job.status = "completed"
        except Exception as exc:
            logger.exception("Matrix job %s failed", job.job_id)
            job.status = "failed"
            job.error = str(exc)

        job.finished_at = time.time()
        job.cost_usd = max(0.0, get_cost_tracker().spent_today() - before)
        return job


def new_job(doc_ids: Sequence[str], clause_slugs: Sequence[str], titles: dict[str, str]) -> MatrixJob:
    return MatrixJob(
        job_id=uuid.uuid4().hex[:12],
        doc_ids=list(doc_ids),
        clause_slugs=list(clause_slugs),
        titles=titles,
    )
