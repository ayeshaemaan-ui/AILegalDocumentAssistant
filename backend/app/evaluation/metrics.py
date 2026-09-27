"""Retrieval metrics (Module 8: LLM Evaluation Metrics + Tier 1 M4: RAG Testing).

These are computed against lawyer-marked spans, so they are objective - no
model judges them. Each answers a different question, which is why reporting
one number is misleading:

  hit_rate@k    Did *any* relevant chunk make the top k? The floor. If this is
                low, nothing downstream can save the answer - the model was
                never shown the clause.
  span_recall   What fraction of the lawyer-marked passages did we retrieve?
                Measured in spans, not chunks - an answer often needs several
                clauses (an obligation plus its carve-out), and this says
                whether we found all of them.
  precision@k   What fraction of what we returned was relevant? Low precision
                means we are burning context tokens on noise.
  MRR           How high did the first relevant chunk rank? Position matters:
                models attend more reliably to what appears early in the prompt.
  nDCG@k        Rank-weighted relevance, discounted logarithmically. The single
                best summary metric when you want one number.

The pair to watch is hit_rate vs MRR. If hit_rate is high but MRR is low, the
retriever is finding the right clause and burying it - that is a re-ranking
problem, not a retrieval problem, and it is the most common failure in a system
like this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(slots=True)
class RetrievalScores:
    hit_rate: float = 0.0
    span_recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    first_relevant_rank: int | None = None
    relevant_chunks: int = 0
    spans_found: int = 0
    spans_total: int = 0
    retrieved: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hit_rate": round(self.hit_rate, 4),
            "span_recall": round(self.span_recall, 4),
            "precision": round(self.precision, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "first_relevant_rank": self.first_relevant_rank,
            "relevant_chunks": self.relevant_chunks,
            "spans_found": self.spans_found,
            "spans_total": self.spans_total,
            "retrieved": self.retrieved,
        }


def score_retrieval(
    relevance_flags: Sequence[bool], spans_found: int, spans_total: int
) -> RetrievalScores:
    """Score one ranked result list.

    Two units of measurement are in play, and conflating them produces
    impossible numbers (recall above 1.0), so they are kept separate:

      * a **gold span** is one lawyer-marked passage. There are `spans_total`
        of them for this question, and recall asks how many were retrieved.
      * a **chunk** is one retrieval result. A single gold span often spills
        across two or three chunks, so the count of relevant *chunks* can
        exceed the count of *spans* - which is why chunk counts must never be
        used as the recall denominator.

    So: `span_recall` measures coverage of the evidence (span units), while
    `precision`, `mrr` and `ndcg` measure the quality of the ranking (chunk
    units). nDCG's ideal ranking is the relevant chunks we actually found,
    placed at the top - that keeps it in [0, 1] and makes it read as "how well
    were the right chunks ranked", which is the question re-ranking answers.
    """
    k = len(relevance_flags)
    relevant_chunks = sum(1 for flag in relevance_flags if flag)
    first_rank = next((i + 1 for i, flag in enumerate(relevance_flags) if flag), None)

    # DCG with binary gains: each relevant chunk contributes 1/log2(rank+1).
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, flag in enumerate(relevance_flags, start=1)
        if flag
    )
    ideal_count = min(relevant_chunks, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return RetrievalScores(
        hit_rate=1.0 if relevant_chunks else 0.0,
        span_recall=spans_found / spans_total if spans_total else 0.0,
        precision=relevant_chunks / k if k else 0.0,
        mrr=1.0 / first_rank if first_rank else 0.0,
        ndcg=dcg / idcg if idcg else 0.0,
        first_relevant_rank=first_rank,
        relevant_chunks=relevant_chunks,
        spans_found=spans_found,
        spans_total=spans_total,
        retrieved=k,
    )


@dataclass(slots=True)
class Aggregate:
    """Mean of a metric across examples, with the spread that mean hides."""

    name: str
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(value)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        idx = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
        return ordered[idx]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 4),
            "p50": round(self.percentile(50), 4),
            "p10": round(self.percentile(10), 4),
            "p90": round(self.percentile(90), 4),
            "n": len(self.values),
        }


def citation_accuracy(answer: str, citations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Did the answer cite excerpts that were actually relevant?

    An answer can be right and still cite badly - pointing at excerpt [3] for a
    fact that came from [1]. This measures whether the citations the model chose
    to use were themselves relevant, which is what a reader relies on when they
    click through to verify.
    """
    import re

    cited = {int(m.group(1)) for m in re.finditer(r"\[(\d{1,2})\]", answer)}
    valid = {c for c in cited if 1 <= c <= len(citations)}
    relevant_cited = sum(
        1 for c in citations if c.get("index") in valid and c.get("is_relevant")
    )
    return {
        "cited_count": len(valid),
        "invalid_citations": sorted(cited - valid),
        "relevant_cited": relevant_cited,
        "citation_precision": round(relevant_cited / len(valid), 4) if valid else 0.0,
        "uncited": len(valid) == 0,
    }
