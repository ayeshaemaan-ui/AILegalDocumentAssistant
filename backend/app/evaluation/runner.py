"""Evaluation runner (Tier 1 M4: RAG Testing, Regression Testing, Eval Frameworks).

Runs the golden set through the system and reports what happened, in two modes:

  retrieval_only   scores retrieval against CUAD's lawyer-marked spans. No
                   generation, so it is free and fast - this is the mode to use
                   while tuning chunking, hybrid weights or re-ranking, because
                   it isolates the variable you are actually changing.
  full             also generates answers and grades them (faithfulness +
                   LLM-as-judge against the lawyer's reference). Costs money;
                   run it when you are ready to measure end to end.

Every run is written to SQLite with the configuration that produced it -
chunking strategy, embedding model, top_k, whether hybrid and re-ranking were
on. That is what turns evaluation into **regression testing**: two runs are
comparable only if you know what differed, and `compare_runs` will tell you
which metrics moved.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from app.config import Settings, get_settings
from app.evaluation.golden_set import (
    GoldenExample,
    chunk_is_relevant,
    get_golden_set,
    span_matches_chunk,
)
from app.evaluation.llm_judge import AnswerJudge, FaithfulnessJudge
from app.evaluation.metrics import Aggregate, citation_accuracy, score_retrieval
from app.retrieval.retriever import RetrievalFilters, get_retriever
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id       TEXT PRIMARY KEY,
    created_at   REAL NOT NULL,
    mode         TEXT NOT NULL,
    sample_size  INTEGER,
    config_json  TEXT,
    metrics_json TEXT,
    elapsed_s    REAL,
    cost_usd     REAL
);
CREATE TABLE IF NOT EXISTS eval_examples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    example_id  TEXT,
    category    TEXT,
    doc_id      TEXT,
    question    TEXT,
    result_json TEXT,
    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_eval_examples_run ON eval_examples(run_id);
"""


@dataclass(slots=True)
class ExampleResult:
    example_id: str
    question: str
    category: str
    doc_id: str
    retrieval: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    faithfulness: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    citations: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "category": self.category,
            "doc_id": self.doc_id,
            "retrieval": self.retrieval,
            "answer": self.answer,
            "faithfulness": self.faithfulness,
            "judge": self.judge,
            "citations": self.citations,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "error": self.error,
        }


class EvaluationRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db_path = self.settings.state_db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def config_snapshot(self) -> dict[str, Any]:
        """Everything that could change a score - the run's fingerprint."""
        s = self.settings
        return {
            "chunk_strategy": s.chunk_strategy,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "embedding_model": s.embedding_model,
            "vector_backend": s.vector_backend,
            "hybrid_enabled": s.hybrid_enabled,
            "dense_weight": s.dense_weight,
            "sparse_weight": s.sparse_weight,
            "rrf_k": s.rrf_k,
            "rerank_enabled": s.rerank_enabled,
            "rerank_backend": s.rerank_backend,
            "mmr_enabled": s.mmr_enabled,
            "retrieval_top_k": s.retrieval_top_k,
            "final_top_k": s.final_top_k,
            "answer_model": s.resolved_answer_model,
            "judge_model": s.resolved_judge_model,
            "answer_effort": s.answer_effort,
        }

    # ------------------------------------------------------------- one row
    async def evaluate_example(
        self, example: GoldenExample, *, retrieval_only: bool, judge: bool
    ) -> ExampleResult:
        started = time.perf_counter()
        result = ExampleResult(
            example_id=example.example_id,
            question=example.question,
            category=example.category,
            doc_id=example.doc_id,
        )

        try:
            # Scope retrieval to the document the question is about. Without
            # this, "what is the governing law?" is ambiguous across 510
            # contracts and the metric would measure nothing.
            chunks, trace = await get_retriever().retrieve(
                example.question,
                filters=RetrievalFilters(doc_ids=[example.doc_id]),
                top_k=self.settings.final_top_k,
            )
        except Exception as exc:
            logger.exception("Retrieval failed for %s", example.example_id)
            result.error = f"retrieval: {exc}"
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        payloads = [c.to_dict() for c in chunks]
        flags = [chunk_is_relevant(p, example) for p in payloads]

        # Span-level coverage: how many of the lawyer-marked passages did at
        # least one retrieved chunk capture? This is the recall denominator -
        # counting relevant chunks instead would let recall exceed 1.0, because
        # one long span routinely lands in several adjacent chunks.
        spans_found = sum(
            1
            for span in example.gold_spans
            if any(
                p.get("doc_id") == example.doc_id and span_matches_chunk(span, p.get("text", ""))
                for p in payloads
            )
        )
        scores = score_retrieval(flags, spans_found, len(example.gold_spans))
        result.retrieval = {
            **scores.as_dict(),
            "reranked": trace.reranked,
            "dense_count": trace.dense_count,
            "sparse_count": trace.sparse_count,
        }

        if retrieval_only:
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        # ---- generate + grade ------------------------------------------
        from app.core.service import get_service

        try:
            envelope = await get_service().answer(
                example.question,
                session_id=None,
                filters={"doc_ids": [example.doc_id]},
                client_key="evaluation",
                persist=False,   # eval runs must not pollute real session history
            )
        except Exception as exc:
            logger.exception("Generation failed for %s", example.example_id)
            result.error = f"generation: {exc}"
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        result.answer = envelope.get("answer", "")
        result.cost_usd = (envelope.get("usage") or {}).get("cost_usd", 0.0)
        result.faithfulness = envelope.get("groundedness", {})

        cited = envelope.get("citations", [])
        for citation in cited:
            citation["is_relevant"] = chunk_is_relevant(
                {"doc_id": citation.get("doc_id"), "text": citation.get("text", "")}, example
            )
        result.citations = citation_accuracy(result.answer, cited)

        if judge and example.reference_answer:
            grade, usage = await AnswerJudge(self.settings).grade(
                question=example.question,
                answer=result.answer,
                reference=example.reference_answer,
                gold_spans=example.gold_spans,
            )
            result.judge = grade.as_dict()
            result.cost_usd += usage.cost_usd

        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    # ----------------------------------------------------------- full run
    async def run(
        self,
        *,
        sample_size: int = 25,
        seed: int = 42,
        categories: Sequence[str] | None = None,
        retrieval_only: bool = False,
        judge: bool = True,
        concurrency: int = 4,
        progress: Callable[[int, int, ExampleResult], None] | None = None,
    ) -> dict[str, Any]:
        run_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()

        golden = get_golden_set(self.settings)
        # Only score documents that are actually indexed - otherwise a run
        # reports 0.0 recall for questions about contracts we never ingested.
        from app.ingestion.pipeline import get_pipeline

        indexed = {d["doc_id"] for d in get_pipeline().manifest.list()}
        if not indexed:
            raise RuntimeError("Nothing is indexed - ingest documents before evaluating.")

        examples = golden.sample(
            sample_size, seed=seed, categories=categories, doc_ids=indexed
        )
        if not examples:
            raise RuntimeError(
                "No golden examples matched the indexed documents. "
                "Ingest CUAD contracts, or widen the category filter."
            )

        semaphore = asyncio.Semaphore(concurrency)
        results: list[ExampleResult] = []
        completed = 0

        async def run_one(example: GoldenExample) -> ExampleResult:
            nonlocal completed
            async with semaphore:
                result = await self.evaluate_example(
                    example, retrieval_only=retrieval_only, judge=judge
                )
            completed += 1
            if progress:
                progress(completed, len(examples), result)
            return result

        results = list(await asyncio.gather(*(run_one(e) for e in examples)))
        elapsed = time.perf_counter() - started
        metrics = self._aggregate(results, retrieval_only=retrieval_only)
        payload = {
            "run_id": run_id,
            "created_at": time.time(),
            "mode": "retrieval_only" if retrieval_only else "full",
            "sample_size": len(results),
            "elapsed_s": round(elapsed, 2),
            "cost_usd": round(sum(r.cost_usd for r in results), 6),
            "config": self.config_snapshot(),
            "metrics": metrics,
            "examples": [r.as_dict() for r in results],
        }
        self._persist(payload)
        return payload

    # ---------------------------------------------------------- aggregate
    def _aggregate(self, results: list[ExampleResult], *, retrieval_only: bool) -> dict[str, Any]:
        buckets = {
            name: Aggregate(name)
            for name in ("hit_rate", "span_recall", "precision", "mrr", "ndcg")
        }
        by_category: dict[str, list[float]] = {}
        latency = Aggregate("latency_ms")
        errors = 0

        for result in results:
            if result.error:
                errors += 1
                continue
            for name, bucket in buckets.items():
                bucket.add(float(result.retrieval.get(name, 0.0)))
            by_category.setdefault(result.category, []).append(
                float(result.retrieval.get("ndcg", 0.0))
            )
            latency.add(result.latency_ms)

        metrics: dict[str, Any] = {
            "retrieval": {name: bucket.as_dict() for name, bucket in buckets.items()},
            "latency_ms": latency.as_dict(),
            "errors": errors,
            "scored": len(results) - errors,
            "by_category_ndcg": {
                category: round(sum(values) / len(values), 4)
                for category, values in sorted(by_category.items())
            },
            "weakest_categories": sorted(
                (
                    {"category": c, "ndcg": round(sum(v) / len(v), 4), "n": len(v)}
                    for c, v in by_category.items()
                ),
                key=lambda entry: entry["ndcg"],
            )[:5],
        }

        if retrieval_only:
            return metrics

        graded = [r for r in results if r.judge and not r.judge.get("failed")]
        if graded:
            metrics["answer_quality"] = {
                "correctness": round(sum(r.judge["correctness"] for r in graded) / len(graded), 3),
                "completeness": round(sum(r.judge["completeness"] for r in graded) / len(graded), 3),
                "citation_quality": round(
                    sum(r.judge["citation_quality"] for r in graded) / len(graded), 3
                ),
                "overall": round(sum(r.judge["overall"] for r in graded) / len(graded), 3),
                "graded": len(graded),
            }

        faithful = [r for r in results if r.faithfulness.get("verdict")]
        if faithful:
            verdicts: dict[str, int] = {}
            for r in faithful:
                verdict = r.faithfulness["verdict"]
                verdicts[verdict] = verdicts.get(verdict, 0) + 1
            ratios = [
                r.faithfulness.get("grounded_ratio", 0.0)
                for r in faithful
                if r.faithfulness.get("grounded_ratio", -1) >= 0
            ]
            metrics["faithfulness"] = {
                "verdicts": verdicts,
                "hallucination_rate": round(
                    verdicts.get("hallucinated", 0) / len(faithful), 4
                ),
                "mean_grounded_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
                "n": len(faithful),
            }

        cited = [r for r in results if r.citations]
        if cited:
            metrics["citations"] = {
                "mean_citation_precision": round(
                    sum(r.citations.get("citation_precision", 0.0) for r in cited) / len(cited), 4
                ),
                "answers_without_citations": sum(1 for r in cited if r.citations.get("uncited")),
                "answers_with_invalid_citations": sum(
                    1 for r in cited if r.citations.get("invalid_citations")
                ),
            }
        return metrics

    # ------------------------------------------------------------ storage
    def _persist(self, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO eval_runs
                   (run_id,created_at,mode,sample_size,config_json,metrics_json,elapsed_s,cost_usd)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    payload["run_id"], payload["created_at"], payload["mode"],
                    payload["sample_size"], json.dumps(payload["config"]),
                    json.dumps(payload["metrics"]), payload["elapsed_s"], payload["cost_usd"],
                ),
            )
            conn.executemany(
                """INSERT INTO eval_examples (run_id,example_id,category,doc_id,question,result_json)
                   VALUES (?,?,?,?,?,?)""",
                [
                    (
                        payload["run_id"], e["example_id"], e["category"], e["doc_id"],
                        e["question"], json.dumps(e, default=str),
                    )
                    for e in payload["examples"]
                ],
            )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            entry = dict(row)
            entry["config"] = json.loads(entry.pop("config_json") or "{}")
            entry["metrics"] = json.loads(entry.pop("metrics_json") or "{}")
            out.append(entry)
        return out

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            examples = conn.execute(
                "SELECT result_json FROM eval_examples WHERE run_id = ?", (run_id,)
            ).fetchall()
        entry = dict(row)
        entry["config"] = json.loads(entry.pop("config_json") or "{}")
        entry["metrics"] = json.loads(entry.pop("metrics_json") or "{}")
        entry["examples"] = [json.loads(e["result_json"]) for e in examples]
        return entry

    def compare_runs(self, baseline_id: str, candidate_id: str) -> dict[str, Any]:
        """Regression testing: what changed, and did it help?"""
        baseline = self.get_run(baseline_id)
        candidate = self.get_run(candidate_id)
        if baseline is None or candidate is None:
            raise ValueError("Both run ids must exist.")

        deltas: dict[str, Any] = {}
        for metric in ("hit_rate", "span_recall", "precision", "mrr", "ndcg"):
            before = baseline["metrics"].get("retrieval", {}).get(metric, {}).get("mean", 0.0)
            after = candidate["metrics"].get("retrieval", {}).get(metric, {}).get("mean", 0.0)
            deltas[metric] = {
                "baseline": before,
                "candidate": after,
                "delta": round(after - before, 4),
                "direction": "better" if after > before else "worse" if after < before else "same",
            }

        config_diff = {
            key: {"baseline": baseline["config"].get(key), "candidate": candidate["config"].get(key)}
            for key in set(baseline["config"]) | set(candidate["config"])
            if baseline["config"].get(key) != candidate["config"].get(key)
        }
        return {
            "baseline_id": baseline_id,
            "candidate_id": candidate_id,
            "config_changed": config_diff,
            "retrieval_deltas": deltas,
            "verdict": _verdict(deltas),
        }


def _verdict(deltas: dict[str, Any]) -> str:
    """nDCG is the headline - it is the only one that accounts for rank."""
    ndcg = deltas.get("ndcg", {}).get("delta", 0.0)
    if ndcg > 0.02:
        return "improvement"
    if ndcg < -0.02:
        return "regression"
    return "no significant change"


_runner: EvaluationRunner | None = None


def get_runner(settings: Settings | None = None) -> EvaluationRunner:
    global _runner
    if _runner is None:
        _runner = EvaluationRunner(settings)
    return _runner
