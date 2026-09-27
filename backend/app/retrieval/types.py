"""Shared retrieval types.

`RetrievedChunk` is the single currency of the retrieval stack: the vector
store emits them, the fuser merges them, the re-ranker re-orders them, and the
prompt builder renders them as citations. Every stage records its own score, so
by the time a chunk reaches the UI you can see exactly why it was chosen -
dense rank, sparse rank, fused score, re-rank score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Per-stage scores, filled in as the chunk moves down the pipeline.
    dense_score: float | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    final_score: float = 0.0
    retriever: str = ""

    @property
    def title(self) -> str:
        return self.metadata.get("title") or self.doc_id

    @property
    def section_path(self) -> str | None:
        return self.metadata.get("section_path")

    @property
    def page_number(self) -> int | None:
        page = self.metadata.get("page_number")
        return int(page) if page is not None else None

    @property
    def page_confidence(self) -> str:
        return self.metadata.get("page_confidence", "estimated")

    def citation_label(self) -> str:
        """Human-readable locator, leading with the section - what a lawyer cites.

        Page is appended only when it is exact (from a real PDF); an estimated
        page is marked with "~" so nobody mistakes a heuristic for a fact.
        """
        parts = [self.title]
        if self.section_path:
            parts.append(f"§ {self.section_path}")
        page = self.page_number
        if page is not None:
            marker = "p." if self.page_confidence == "exact" else "~p."
            parts.append(f"{marker}{page}")
        return " · ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "citation": self.citation_label(),
            "scores": {
                "dense": self.dense_score,
                "dense_rank": self.dense_rank,
                "sparse": self.sparse_score,
                "sparse_rank": self.sparse_rank,
                "fused": self.fused_score,
                "rerank": self.rerank_score,
                "final": self.final_score,
            },
            "retriever": self.retriever,
        }
