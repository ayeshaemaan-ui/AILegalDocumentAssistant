"""The retrieval pipeline (Module 5: Retrieval Pipeline).

One class, `LegalRetriever`, runs the full funnel and records what it did at
each stage so the UI and the evaluation harness can both inspect it:

    question
      -> metadata filter        narrow the search space (doc / clause / law)
      -> dense search  (k=20)   semantic recall
      -> sparse search (k=20)   exact-term recall
      -> RRF fusion             one ranked list from two
      -> MMR                    drop near-duplicates
      -> cross-encoder rerank   precision pass over the shortlist
      -> top 6                  what actually enters the prompt

Retrieving 20+20 and returning 6 is deliberate. Recall is cheap at the vector
level and expensive in the prompt: every extra chunk costs tokens, dilutes
attention, and gives the model more opportunity to cite the wrong clause.

`LangChainLegalRetriever` at the bottom wraps this as a LangChain
`BaseRetriever` so it composes with LCEL chains and LangChain tools.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.core.embeddings import get_embeddings
from app.retrieval.bm25 import get_bm25_index
from app.retrieval.hybrid import maximal_marginal_relevance, reciprocal_rank_fusion
from app.retrieval.rerank import get_reranker
from app.retrieval.types import RetrievedChunk
from app.retrieval.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalFilters:
    """Metadata filters (Tier 1 M1: Metadata Filtering)."""

    doc_ids: list[str] = field(default_factory=list)
    clause_types: list[str] = field(default_factory=list)
    contract_types: list[str] = field(default_factory=list)
    governing_law: str | None = None

    def to_where(self) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if self.doc_ids:
            clauses.append({"doc_id": {"$in": self.doc_ids}})
        if self.contract_types:
            clauses.append({"contract_type": {"$in": self.contract_types}})
        if self.governing_law:
            clauses.append({"governing_law": self.governing_law})
        if self.clause_types:
            # clause_types_csv is stored as ",a,b,c," so $contains ",a," is an
            # exact tag match, not a prefix collision.
            tag_clauses = [
                {"clause_types_csv": {"$contains": f",{slug},"}} for slug in self.clause_types
            ]
            clauses.append(tag_clauses[0] if len(tag_clauses) == 1 else {"$or": tag_clauses})

        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def is_empty(self) -> bool:
        return not (self.doc_ids or self.clause_types or self.contract_types or self.governing_law)


@dataclass(slots=True)
class RetrievalTrace:
    """Per-stage diagnostics - surfaced in the UI's 'retrieval' panel."""

    query: str
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    after_mmr: int = 0
    final_count: int = 0
    reranked: bool = False
    filters_applied: dict[str, Any] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    fallback_used: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "dense_count": self.dense_count,
            "sparse_count": self.sparse_count,
            "fused_count": self.fused_count,
            "after_mmr": self.after_mmr,
            "final_count": self.final_count,
            "reranked": self.reranked,
            "filters_applied": self.filters_applied,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "fallback_used": self.fallback_used,
        }


class LegalRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embeddings = get_embeddings(self.settings)
        self.store = get_vector_store(self.settings)
        self.bm25 = get_bm25_index()
        self.reranker = get_reranker(self.settings)

    # ------------------------------------------------------------ indexing
    def refresh_bm25(self) -> int:
        """Rebuild the sparse index from whatever is in the vector store."""
        try:
            chunks = self.store.all_chunks()
        except NotImplementedError:
            logger.warning("Backend has no full scan; sparse retrieval disabled.")
            return 0
        return self.bm25.build(chunks)

    # ----------------------------------------------------------- retrieval
    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        settings = self.settings
        top_k = top_k or settings.final_top_k
        candidate_k = candidate_k or settings.retrieval_top_k
        filters = filters or RetrievalFilters()
        where = filters.to_where()
        trace = RetrievalTrace(query=query, filters_applied=where)

        # -- dense ---------------------------------------------------------
        t0 = time.perf_counter()
        query_vector = await self.embeddings.aembed_query(query)
        dense = await asyncio.to_thread(self.store.search, query_vector, candidate_k, where)
        trace.timings_ms["dense"] = (time.perf_counter() - t0) * 1000
        trace.dense_count = len(dense)

        # A filter that matches nothing is worse than no filter - the user gets
        # "I don't know" for a question the corpus can answer. Retry unfiltered
        # and tell the trace, so the UI can say the filter was relaxed.
        if not dense and where is not None:
            dense = await asyncio.to_thread(self.store.search, query_vector, candidate_k, None)
            trace.fallback_used = "filter_relaxed"
            trace.dense_count = len(dense)
            where = None

        # -- sparse --------------------------------------------------------
        sparse: list[RetrievedChunk] = []
        if settings.hybrid_enabled and self.bm25.size:
            t1 = time.perf_counter()
            sparse = await asyncio.to_thread(self.bm25.search, query, candidate_k, where)
            trace.timings_ms["sparse"] = (time.perf_counter() - t1) * 1000
            trace.sparse_count = len(sparse)

        # -- fuse ----------------------------------------------------------
        if sparse:
            candidates = reciprocal_rank_fusion(
                dense,
                sparse,
                k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
            )
        else:
            candidates = list(dense)
            for chunk in candidates:
                chunk.final_score = chunk.dense_score or 0.0
        trace.fused_count = len(candidates)

        if not candidates:
            trace.final_count = 0
            return [], trace

        # -- diversity -----------------------------------------------------
        if settings.mmr_enabled and len(candidates) > top_k:
            t2 = time.perf_counter()
            pool = candidates[: max(top_k * 4, candidate_k)]
            vectors = await self.embeddings.aembed_documents([c.text for c in pool])
            keep = max(top_k, top_k * 2 if self.reranker else top_k)
            candidates = maximal_marginal_relevance(
                pool, query_vector, vectors, k=keep, lambda_mult=settings.mmr_lambda
            )
            trace.timings_ms["mmr"] = (time.perf_counter() - t2) * 1000
        trace.after_mmr = len(candidates)

        # -- rerank --------------------------------------------------------
        if self.reranker is not None:
            t3 = time.perf_counter()
            candidates = await self.reranker.arerank(query, candidates, top_k)
            trace.timings_ms["rerank"] = (time.perf_counter() - t3) * 1000
            trace.reranked = True
        else:
            candidates = candidates[:top_k]

        if settings.min_relevance_score > 0:
            filtered = [c for c in candidates if c.final_score >= settings.min_relevance_score]
            # Never return nothing purely on a threshold - a weak answer with
            # visible low confidence beats a blank screen.
            candidates = filtered or candidates[:1]

        trace.final_count = len(candidates)
        return candidates, trace


_retriever: LegalRetriever | None = None


def get_retriever(settings: Settings | None = None) -> LegalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LegalRetriever(settings)
    return _retriever


def reset_retriever() -> None:
    global _retriever
    _retriever = None


# ------------------------------------------------------ LangChain adapter
def build_langchain_retriever(settings: Settings | None = None):
    """Expose the pipeline as a LangChain `BaseRetriever` (Module 6: Retrievers).

    Defined inside a function so importing this module never hard-requires
    LangChain - the FastAPI path uses `LegalRetriever` directly.
    """
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class LangChainLegalRetriever(BaseRetriever):
        """Hybrid + re-ranked retrieval, usable in any LCEL chain or tool."""

        k: int = 6

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
        ) -> list[Document]:
            retriever = get_retriever(settings)
            chunks, _trace = asyncio.run(retriever.retrieve(query, top_k=self.k))
            return [
                Document(
                    page_content=c.text,
                    metadata={**c.metadata, "citation": c.citation_label(), "score": c.final_score},
                )
                for c in chunks
            ]

        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
        ) -> list[Document]:
            retriever = get_retriever(settings)
            chunks, _trace = await retriever.retrieve(query, top_k=self.k)
            return [
                Document(
                    page_content=c.text,
                    metadata={**c.metadata, "citation": c.citation_label(), "score": c.final_score},
                )
                for c in chunks
            ]

    return LangChainLegalRetriever()
