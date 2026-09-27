"""Embeddings (Shared Module 4: Embeddings & Vector Search).

An embedding turns a clause into a point in 768-dimensional space where
"distance" means "difference in meaning". That is the whole trick behind
semantic search: "can I quit early?" lands near a *Termination for Convenience*
clause even though they share no keywords.

We use BAAI/bge-base-en-v1.5 locally:
  * free and offline, so re-indexing 510 contracts while tuning chunking costs
    nothing and hits no rate limit;
  * trained with an asymmetric prefix - queries get an instruction prefix,
    documents do not. Applying the prefix to both sides measurably hurts recall,
    which is why `embed_query` and `embed_documents` are deliberately different.
"""
from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np

from app.config import Settings, get_settings
from app.core.torch_lock import TORCH_INFERENCE_LOCK

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity = dot product of L2-normalised vectors.

    Cosine is used instead of Euclidean distance because it measures *angle*,
    not magnitude: a one-line clause and a three-paragraph clause about the same
    obligation should score as similar, and only cosine ignores the length gap.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


class LocalEmbeddings:
    """sentence-transformers BGE wrapper, shaped like LangChain's `Embeddings`.

    Implementing `embed_documents` / `embed_query` means this object drops
    straight into any LangChain vector store without an adapter.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._lock = threading.Lock()  # guards lazy model construction

    def _ensure_model(self):
        # Lazy + locked: the model is ~440MB and FastAPI workers would otherwise
        # each try to load it on the first concurrent request.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self.settings.embedding_model)
                    self._model = SentenceTransformer(
                        self.settings.embedding_model, device=_pick_device()
                    )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        # Serialised against every other forward pass in the process; see
        # app/core/torch_lock.py for why the lock has to be global.
        with TORCH_INFERENCE_LOCK:
            vectors = model.encode(
            list(texts),
            batch_size=self.settings.embedding_batch_size,
                normalize_embeddings=True,  # pre-normalised => cosine is a dot product
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        prefixed = f"{self.settings.embedding_query_prefix}{text}"
        with TORCH_INFERENCE_LOCK:
            vector = model.encode(
                [prefixed], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
            )
        return vector[0].tolist()

    # Async shims so FastAPI handlers never block the event loop on CPU work.
    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        import asyncio

        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        import asyncio

        return await asyncio.to_thread(self.embed_query, text)


def _pick_device() -> str:
    """Apple Silicon GPU when present - roughly 4x faster than CPU for BGE."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch missing or partially built
        pass
    return "cpu"


_embeddings_singleton: LocalEmbeddings | None = None


def get_embeddings(settings: Settings | None = None) -> LocalEmbeddings:
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = LocalEmbeddings(settings)
    return _embeddings_singleton
