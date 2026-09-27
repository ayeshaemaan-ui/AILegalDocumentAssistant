"""Ingestion pipeline - the offline half of RAG (Module 5: RAG Architecture).

RAG has two halves that run at completely different times. This is the slow
one, and everything expensive belongs here: parsing, chunking, clause tagging,
embedding. The query path should never do work that could have been done once
at ingest.

    file -> load -> chunk -> enrich metadata -> embed -> upsert -> manifest

The manifest (a small SQLite table) records what is indexed and how, so the app
can list documents, delete one cleanly, and tell you which chunking strategy
produced the current index when an evaluation result looks surprising.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from app.config import Settings, get_settings
from app.core.embeddings import get_embeddings
from app.ingestion.chunking import Chunk, chunk_document
from app.ingestion.loaders import LoadedDocument, load_document
from app.ingestion.metadata import build_document_metadata, enrich_chunks
from app.retrieval.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source_path     TEXT,
    file_type       TEXT,
    contract_type   TEXT,
    governing_law   TEXT,
    parties         TEXT,
    page_count      INTEGER,
    char_count      INTEGER,
    chunk_count     INTEGER,
    chunk_strategy  TEXT,
    embedding_model TEXT,
    indexed_at      REAL,
    metadata_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
"""


@dataclass(slots=True)
class IngestResult:
    doc_id: str
    title: str
    chunk_count: int
    strategy: str
    elapsed_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "chunk_count": self.chunk_count,
            "strategy": self.strategy,
            "elapsed_s": round(self.elapsed_s, 2),
            "skipped": self.skipped,
            "error": self.error,
            "metadata": self.metadata,
        }


class Manifest:
    """Tiny SQLite catalogue of what has been indexed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(MANIFEST_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, doc: LoadedDocument, meta: dict[str, Any], chunks: Sequence[Chunk], settings: Settings) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents
                   (doc_id,title,source_path,file_type,contract_type,governing_law,parties,
                    page_count,char_count,chunk_count,chunk_strategy,embedding_model,indexed_at,metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc.doc_id, doc.title, doc.source_path, doc.file_type,
                    meta.get("contract_type"), meta.get("governing_law"),
                    json.dumps(meta.get("parties", [])),
                    meta.get("page_count", 0), meta.get("char_count", 0),
                    len(chunks), settings.chunk_strategy, settings.embedding_model,
                    time.time(), json.dumps(meta, default=str),
                ),
            )

    def has(self, doc_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone() is not None

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY indexed_at DESC"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            entry = dict(row)
            entry["parties"] = json.loads(entry.get("parties") or "[]")
            entry.pop("metadata_json", None)
            out.append(entry)
        return out

    def get(self, doc_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        entry = dict(row)
        entry["parties"] = json.loads(entry.get("parties") or "[]")
        entry["metadata"] = json.loads(entry.get("metadata_json") or "{}")
        entry.pop("metadata_json", None)
        return entry

    def delete(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embeddings = get_embeddings(self.settings)
        self.store = get_vector_store(self.settings)
        self.manifest = Manifest(self.settings.state_db_path)

    def ingest_file(
        self, path: Path | str, *, strategy: str | None = None, force: bool = False
    ) -> IngestResult:
        started = time.perf_counter()
        path = Path(path)
        try:
            doc = load_document(path)
        except Exception as exc:
            logger.warning("Load failed for %s: %s", path.name, exc)
            return IngestResult("", path.stem, 0, strategy or self.settings.chunk_strategy,
                                time.perf_counter() - started, error=str(exc))

        # doc_id is a content hash, so re-uploading the same file is a no-op
        # rather than a silent duplicate that skews every future retrieval.
        if not force and self.manifest.has(doc.doc_id):
            return IngestResult(doc.doc_id, doc.title, 0, strategy or self.settings.chunk_strategy,
                                time.perf_counter() - started, skipped=True)

        return self._ingest_loaded(doc, strategy=strategy, started=started)

    def ingest_text(
        self, title: str, text: str, *, source: str = "inline", strategy: str | None = None
    ) -> IngestResult:
        from app.ingestion.loaders import make_doc_id, normalise_text, _estimate_pages

        started = time.perf_counter()
        body = normalise_text(text)
        doc = LoadedDocument(
            doc_id=make_doc_id(title, body), title=title, source_path=source,
            file_type="text", full_text=body, pages=_estimate_pages(body),
        )
        return self._ingest_loaded(doc, strategy=strategy, started=started)

    def _ingest_loaded(
        self, doc: LoadedDocument, *, strategy: str | None, started: float
    ) -> IngestResult:
        strategy = strategy or self.settings.chunk_strategy
        doc_meta = build_document_metadata(doc)
        chunks = chunk_document(doc, self.settings, strategy=strategy)
        if not chunks:
            return IngestResult(doc.doc_id, doc.title, 0, strategy,
                                time.perf_counter() - started, error="no chunks produced")
        chunks = enrich_chunks(chunks, doc_meta)

        # Replace-in-place: drop any previous vectors for this doc first so a
        # re-index with a different strategy cannot leave orphans behind.
        self.store.delete_document(doc.doc_id)

        batch = self.settings.embedding_batch_size * 4
        total = 0
        for start in range(0, len(chunks), batch):
            window = chunks[start : start + batch]
            vectors = self.embeddings.embed_documents([c.text for c in window])
            total += self.store.upsert(window, vectors)

        self.manifest.record(doc, doc_meta, chunks, self.settings)
        elapsed = time.perf_counter() - started
        logger.info("Indexed %s: %d chunks in %.1fs", doc.title[:60], total, elapsed)
        return IngestResult(doc.doc_id, doc.title, total, strategy, elapsed, metadata=doc_meta)

    def ingest_directory(
        self,
        directory: Path | str,
        *,
        limit: int | None = None,
        patterns: Sequence[str] = ("*.txt", "*.pdf", "*.docx"),
        strategy: str | None = None,
        force: bool = False,
        progress: Callable[[int, int, IngestResult], None] | None = None,
    ) -> list[IngestResult]:
        directory = Path(directory)
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(directory.rglob(pattern)))
        files = sorted(set(files))[: limit or len(files)]

        results: list[IngestResult] = []
        for idx, path in enumerate(files, start=1):
            result = self.ingest_file(path, strategy=strategy, force=force)
            results.append(result)
            if progress:
                progress(idx, len(files), result)
        return results

    def delete_document(self, doc_id: str) -> int:
        removed = self.store.delete_document(doc_id)
        self.manifest.delete(doc_id)
        return removed

    def stats(self) -> dict[str, Any]:
        return {
            "documents": self.manifest.count(),
            "chunks": self.store.count(),
            "chunk_strategy": self.settings.chunk_strategy,
            "embedding_model": self.settings.embedding_model,
            "vector_backend": self.settings.vector_backend,
        }


_pipeline: IngestionPipeline | None = None


def get_pipeline(settings: Settings | None = None) -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(settings)
    return _pipeline
