"""Re-ranking (Module 5: Re-ranking).

Retrieval and re-ranking answer different questions, which is why doing both
beats doing either well.

The embedding model is a *bi-encoder*: it encodes the query and the clause
separately and compares vectors. That is what makes it fast enough to search
40,000 chunks - but it never sees the two texts together, so it cannot notice
that the clause says "Supplier may terminate" when the question asked what
*Customer* may do.

A *cross-encoder* reads query and clause in one forward pass and scores the
pair directly. Far more accurate, far too slow for the whole corpus - so it
runs only over the ~20 candidates hybrid search already shortlisted. That is
the standard retrieve-wide-then-rerank-narrow shape.

Two backends:
  cross_encoder  local ms-marco MiniLM. Free, ~10ms for 20 pairs, no network.
  llm            Claude Haiku scoring each candidate with structured output.
                 Better on subtle legal phrasing, but costs money and latency -
                 worth it when precision matters more than speed.
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
from typing import Sequence

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.core.torch_lock import TORCH_INFERENCE_LOCK
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Logit -> (0, 1), written to not overflow on large-magnitude inputs."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class CrossEncoderReranker:
    """Local cross-encoder over (query, chunk) pairs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._lock = threading.Lock()  # guards lazy model construction

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    from app.core.embeddings import _pick_device

                    logger.info("Loading re-ranker %s", self.settings.rerank_model)
                    self._model = CrossEncoder(self.settings.rerank_model, device=_pick_device())
        return self._model

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self._ensure_model()
        # Truncate to the model's window; the decisive language in a clause is
        # almost always in its opening sentences.
        pairs = [(query, chunk.text[:2000]) for chunk in chunks]
        # Shares one lock with the embedding model - both run on the same GPU.
        with TORCH_INFERENCE_LOCK:
            scores = model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            # ms-marco cross-encoders emit raw logits, roughly -11..+11, so a
            # perfectly good match can score -3.8. Displaying that to a user is
            # meaningless. A sigmoid maps it to (0, 1) as a relevance
            # probability, and being monotonic it leaves the ranking untouched.
            chunk.rerank_score = _sigmoid(float(score))
            chunk.final_score = chunk.rerank_score
        ranked = sorted(chunks, key=lambda c: -(c.rerank_score or 0.0))
        return ranked[:top_k]

    async def arerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.rerank, query, chunks, top_k)


class _Relevance(BaseModel):
    """Structured output schema - the model cannot reply with prose."""

    chunk_index: int = Field(description="Index of the passage being scored.")
    score: int = Field(ge=0, le=10, description="0 = irrelevant, 10 = directly answers.")
    reason: str = Field(max_length=200, description="One short sentence of justification.")


class _RelevanceBatch(BaseModel):
    rankings: list[_Relevance]


LLM_RERANK_SYSTEM = """You are a legal research assistant scoring how well each \
passage answers a question about a contract.

Score each passage 0-10:
  9-10  directly states the answer
  6-8   contains part of the answer or a governing condition
  3-5   related topic but does not answer
  0-2   irrelevant

Judge only what the passage actually says. A passage that discusses the right \
topic but assigns the obligation to a different party is not a good answer - \
score it low. Return a score for every passage index you are given."""


class LLMReranker:
    """Claude-as-re-ranker, using structured output so results are parseable."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def arerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        from app.core.llm import get_llm_client

        passages = "\n\n".join(
            f"[{i}] ({c.citation_label()})\n{c.text[:1200]}" for i, c in enumerate(chunks)
        )
        user = f"Question: {query}\n\nPassages:\n{passages}"

        client = get_llm_client(self.settings)
        try:
            batch, _usage = await client.structured(
                system=LLM_RERANK_SYSTEM,
                messages=[{"role": "user", "content": user}],
                schema=_RelevanceBatch,
                model=self.settings.resolved_fast_model,
                effort="low",
            )
        except Exception as exc:  # never let re-ranking break the answer path
            logger.warning("LLM re-rank failed (%s); keeping fusion order", exc)
            return list(chunks[:top_k])

        by_index = {r.chunk_index: r for r in batch.rankings}
        for idx, chunk in enumerate(chunks):
            ranking = by_index.get(idx)
            chunk.rerank_score = float(ranking.score) / 10.0 if ranking else 0.0
            chunk.final_score = chunk.rerank_score
            if ranking:
                chunk.metadata = {**chunk.metadata, "rerank_reason": ranking.reason}
        return sorted(chunks, key=lambda c: -(c.rerank_score or 0.0))[:top_k]


_cross_encoder: CrossEncoderReranker | None = None


def get_reranker(settings: Settings | None = None):
    """Return the configured re-ranker, or None when re-ranking is disabled."""
    global _cross_encoder
    settings = settings or get_settings()
    if not settings.rerank_enabled or settings.rerank_backend == "none":
        return None
    if settings.rerank_backend == "llm":
        return LLMReranker(settings)
    if _cross_encoder is None:
        _cross_encoder = CrossEncoderReranker(settings)
    return _cross_encoder
