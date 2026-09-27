"""Hybrid search (Tier 1 M1: Hybrid Search).

Two retrievers return two ranked lists whose scores are not comparable - a
cosine similarity of 0.82 and a BM25 score of 14.3 mean nothing to each other.
Reciprocal Rank Fusion sidesteps the problem entirely by throwing away the
scores and fusing on *rank*:

    RRF(chunk) = sum over retrievers of  weight / (k + rank_in_that_list)

A chunk ranked #1 by one arm and #30 by the other still beats a chunk ranked
#15 by both, which is exactly the behaviour we want: agreement is good, but a
strong signal from either arm should survive. `k` (default 60) flattens the
curve so the top few ranks do not completely dominate.

After fusion, MMR trims for diversity. Contracts repeat themselves - the same
indemnity language appears in three adjacent sub-clauses - and without MMR the
6 chunks we inject into the prompt can be 6 near-duplicates of one paragraph,
wasting the context window and hiding the clause that actually answers.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense: Sequence[RetrievedChunk],
    sparse: Sequence[RetrievedChunk],
    *,
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[RetrievedChunk]:
    """Fuse two ranked lists into one, de-duplicating by chunk_id."""
    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for rank, chunk in enumerate(dense, start=1):
        merged[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + dense_weight / (k + rank)

    for rank, chunk in enumerate(sparse, start=1):
        existing = merged.get(chunk.chunk_id)
        if existing is None:
            merged[chunk.chunk_id] = chunk
        else:
            # Same chunk found by both arms - keep both provenance scores so the
            # UI can show *why* it ranked where it did.
            existing.sparse_score = chunk.sparse_score
            existing.sparse_rank = chunk.sparse_rank
            existing.retriever = "hybrid"
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + sparse_weight / (k + rank)

    fused: list[RetrievedChunk] = []
    for chunk_id, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        chunk = merged[chunk_id]
        chunk.fused_score = score
        chunk.final_score = score
        if chunk.retriever != "hybrid":
            chunk.retriever = "dense" if chunk.dense_rank else "sparse"
        fused.append(chunk)
    return fused


def maximal_marginal_relevance(
    chunks: Sequence[RetrievedChunk],
    query_vector: Sequence[float],
    chunk_vectors: Sequence[Sequence[float]],
    *,
    k: int,
    lambda_mult: float = 0.7,
) -> list[RetrievedChunk]:
    """Greedy MMR: relevance to the query, penalised by similarity to what is
    already selected.

        score = lambda * sim(chunk, query) - (1 - lambda) * max sim(chunk, selected)

    lambda_mult=1.0 is pure relevance; 0.0 is pure diversity. 0.7 keeps the
    ranking relevance-led while breaking up near-duplicate runs.
    """
    if not chunks or k <= 0:
        return []
    if len(chunks) <= k:
        return list(chunks)

    query = np.asarray(query_vector, dtype=np.float32)
    matrix = np.asarray(chunk_vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        logger.warning("MMR vector/chunk mismatch - returning relevance order")
        return list(chunks[:k])

    # Vectors arrive L2-normalised from BGE, so a dot product is the cosine.
    query = query / (np.linalg.norm(query) + 1e-10)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    relevance = matrix @ query

    selected: list[int] = [int(np.argmax(relevance))]
    candidates = set(range(len(chunks))) - set(selected)

    while len(selected) < k and candidates:
        best_idx, best_score = None, -np.inf
        selected_matrix = matrix[selected]
        for idx in candidates:
            redundancy = float(np.max(selected_matrix @ matrix[idx]))
            score = lambda_mult * float(relevance[idx]) - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_idx, best_score = idx, score
        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.discard(best_idx)

    return [chunks[i] for i in selected]
