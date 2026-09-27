"""Sparse retrieval (Module 4: Dense vs Sparse Retrieval).

Dense embeddings are good at meaning and bad at *identifiers*. In legal text
the identifiers are often the whole question: a party name, "Section 12.3",
"$5,000,000", "Delaware", "CyberKnife". Those are rare tokens, and a 768-dim
vector smooths them into the surrounding semantics - ask for "Siemens" and
dense retrieval happily returns a distributor clause naming someone else.

BM25 has the opposite bias: it scores exact term overlap, weighting rare terms
heavily (IDF) and damping repetition (term-frequency saturation). Running both
and fusing them is the whole argument for hybrid search - each arm covers the
other's blind spot.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any, Sequence

from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

# Keep money, section numbers and percentages intact as single tokens - they
# carry most of the signal in a contract and a naive \w+ split destroys them.
TOKEN_RE = re.compile(r"\$?\d+(?:[.,]\d+)*%?|[A-Za-z][A-Za-z\-']+")

STOPWORDS = {
    "the","a","an","and","or","of","to","in","for","on","by","with","as","at","from","that",
    "this","these","those","is","are","was","were","be","been","being","shall","will","may",
    "any","all","such","it","its","their","his","her","which","who","whom","not","no",
    "if","then","than","so","but","also","have","has","had","do","does","did","hereby","herein",
    "hereof","hereto","thereof","therein","said","other","each","per","upon","under","into",
}


def tokenize(text: str) -> list[str]:
    return [
        t.lower()
        for t in TOKEN_RE.findall(text)
        if len(t) > 1 and t.lower() not in STOPWORDS
    ]


class BM25Index:
    """In-memory BM25 over the whole corpus.

    Rebuilt whenever documents are added or removed. For 510 contracts (~40k
    chunks) this is a few hundred MB and rebuilds in seconds, which is a fine
    trade for exact-term recall. At a much larger scale this would move into
    Postgres full-text or Weaviate's native hybrid.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: list[RetrievedChunk] = []
        self._token_sets: list[set[str]] = []
        self._lock = threading.Lock()

    def build(self, chunks: Sequence[RetrievedChunk]) -> int:
        from rank_bm25 import BM25Okapi

        with self._lock:
            self._chunks = list(chunks)
            corpus = [tokenize(c.text) for c in self._chunks]
            # Keep our own token sets: they are what decide whether a chunk is a
            # candidate at all (see `search`), independent of BM25's scoring.
            self._token_sets = [set(tokens) for tokens in corpus]
            # BM25Okapi rejects an empty corpus, so guard the cold-start case.
            self._bm25 = BM25Okapi(corpus) if corpus else None
            logger.info("BM25 index built over %d chunks", len(self._chunks))
            return len(self._chunks)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)

        # Candidacy is decided by actual lexical overlap, NOT by `score > 0`.
        # BM25's IDF term goes negative for a token that appears in every
        # document, so a positive-score filter silently returns nothing for the
        # most common legal terms on a small corpus - the exact case where the
        # sparse arm is supposed to help.
        candidates = [
            (score, chunk)
            for score, chunk, tokens in zip(scores, self._chunks, self._token_sets)
            if (query_set & tokens) and _matches(_filterable(chunk), where)
        ]
        candidates.sort(key=lambda t: -t[0])
        top = candidates[:k]
        if not top:
            return []

        # Min-max normalise so the score reads as 0-1 and is comparable across
        # queries; raw BM25 magnitudes depend on query length, and can be
        # negative, so dividing by the best score alone is not enough.
        highest = top[0][0]
        lowest = top[-1][0]
        span = highest - lowest

        out: list[RetrievedChunk] = []
        for rank, (score, chunk) in enumerate(top):
            normalised = 1.0 if span <= 0 else (score - lowest) / span
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    sparse_score=float(normalised),
                    sparse_rank=rank + 1,
                    retriever="sparse",
                )
            )
        return out


def _filterable(chunk: RetrievedChunk) -> dict[str, Any]:
    """Metadata view used for filtering.

    `doc_id` is a first-class field on RetrievedChunk and is *usually* also
    mirrored into metadata by the vector store, but a filter that silently
    returns nothing when it is not mirrored is a nasty failure mode - the user
    sees "no results" for a document that is definitely indexed. Fold the
    field in explicitly rather than depending on the duplication.
    """
    if chunk.doc_id and "doc_id" not in chunk.metadata:
        return {**chunk.metadata, "doc_id": chunk.doc_id}
    return chunk.metadata


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Apply the same metadata filters the vector store would.

    Supports the subset of Chroma's `where` grammar this app actually emits:
    equality, `$in`, `$contains`, and a top-level `$or`.
    """
    if not where:
        return True
    if "$or" in where:
        return any(_matches(metadata, clause) for clause in where["$or"])
    if "$and" in where:
        return all(_matches(metadata, clause) for clause in where["$and"])

    for key, condition in where.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition and value not in condition["$in"]:
                return False
            if "$contains" in condition and condition["$contains"] not in str(value or ""):
                return False
            if "$eq" in condition and value != condition["$eq"]:
                return False
            if "$ne" in condition and value == condition["$ne"]:
                return False
        elif value != condition:
            return False
    return True


_index_singleton: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _index_singleton
    if _index_singleton is None:
        _index_singleton = BM25Index()
    return _index_singleton
