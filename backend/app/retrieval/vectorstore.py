"""Vector store layer (Module 4: Vector DBs - Chroma / PGVector / Pinecone / Weaviate).

All four backends sit behind one `VectorStore` interface. Chroma is the default
because it is embedded (no server, no network, no cost) which matters when you
are re-indexing 510 contracts repeatedly to tune chunking. The other three are
real adapters, not stubs - switch with `VECTOR_BACKEND` in .env - and their
imports are lazy so the app never requires a driver it will not use.

What differs between them in practice is operational, not semantic:
  Chroma    embedded, single-file persistence. Best for local + CI.
  PGVector  rides your existing Postgres; transactional with your app data.
  Pinecone  managed and serverless; no ops, pay per read/write.
  Weaviate  self-hostable, native hybrid (BM25 + vector) inside the engine.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

from app.config import Settings, get_settings
from app.ingestion.chunking import Chunk
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int: ...

    @abstractmethod
    def search(
        self, vector: Sequence[float], k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> int: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def all_chunks(self) -> list[RetrievedChunk]: ...


class ChromaStore(VectorStore):
    """Embedded Chroma - the default backend."""

    def __init__(self, settings: Settings | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings or get_settings()
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # Cosine, to match the L2-normalised BGE vectors we produce.
        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )
        return len(chunks)

    def search(
        self, vector: Sequence[float], k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=max(1, k),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        out: list[RetrievedChunk] = []
        for rank, (cid, text, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
            meta = dict(meta or {})
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id", ""),
                    text=text or "",
                    metadata=meta,
                    # Chroma returns cosine *distance*; similarity is 1 - d.
                    dense_score=1.0 - float(dist),
                    dense_rank=rank + 1,
                    retriever="dense",
                )
            )
        return out

    def delete_document(self, doc_id: str) -> int:
        existing = self._collection.get(where={"doc_id": doc_id}, include=[])
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self._collection.count()

    # Chroma's SQLite backend binds one variable per row it returns, so an
    # unbounded `get()` trips SQLITE_MAX_VARIABLE_NUMBER (32,766) once the
    # collection outgrows it - which the 510-contract corpus does at ~35k
    # chunks. Page through instead; the limit is per statement, not per scan.
    _SCAN_PAGE = 5_000

    def _scan(self, include: list[str]) -> Iterator[dict[str, Any]]:
        """Yield every record in the collection, one page at a time."""
        offset = 0
        while True:
            page = self._collection.get(
                include=include, limit=self._SCAN_PAGE, offset=offset
            )
            ids = page.get("ids") or []
            if not ids:
                return
            yield page
            if len(ids) < self._SCAN_PAGE:
                return
            offset += len(ids)

    def all_chunks(self) -> list[RetrievedChunk]:
        """Full dump - used to build the in-memory BM25 index at startup."""
        out: list[RetrievedChunk] = []
        for payload in self._scan(["documents", "metadatas"]):
            for cid, text, meta in zip(
                payload.get("ids", []),
                payload.get("documents", []),
                payload.get("metadatas", []),
            ):
                meta = dict(meta or {})
                out.append(
                    RetrievedChunk(
                        chunk_id=cid, doc_id=meta.get("doc_id", ""), text=text or "", metadata=meta
                    )
                )
        return out

    def list_documents(self) -> list[dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        for meta in (m for page in self._scan(["metadatas"]) for m in page.get("metadatas", [])):
            meta = dict(meta or {})
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue
            entry = docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "title": meta.get("title", doc_id),
                    "contract_type": meta.get("contract_type"),
                    "governing_law": meta.get("governing_law"),
                    "party_primary": meta.get("party_primary"),
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1
        return sorted(docs.values(), key=lambda d: d["title"].lower())

    def reset(self) -> None:
        self._client.delete_collection(self.settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name, metadata={"hnsw:space": "cosine"}
        )


class PGVectorStore(VectorStore):
    """Postgres + pgvector, via langchain-postgres."""

    def __init__(self, settings: Settings | None = None) -> None:
        from langchain_postgres import PGVector

        from app.core.embeddings import get_embeddings

        self.settings = settings or get_settings()
        self._store = PGVector(
            embeddings=get_embeddings(self.settings),
            collection_name=self.settings.collection_name,
            connection=self.settings.pg_connection,
            use_jsonb=True,
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        from langchain_core.documents import Document

        self._store.add_documents(
            [Document(page_content=c.text, metadata=c.to_metadata()) for c in chunks],
            ids=[c.chunk_id for c in chunks],
        )
        return len(chunks)

    def search(
        self, vector: Sequence[float], k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        hits = self._store.similarity_search_with_score_by_vector(list(vector), k=k, filter=where)
        return [
            RetrievedChunk(
                chunk_id=doc.metadata.get("chunk_id", ""),
                doc_id=doc.metadata.get("doc_id", ""),
                text=doc.page_content,
                metadata=dict(doc.metadata),
                dense_score=1.0 - float(score),
                dense_rank=rank + 1,
                retriever="dense",
            )
            for rank, (doc, score) in enumerate(hits)
        ]

    def delete_document(self, doc_id: str) -> int:
        self._store.delete(filter={"doc_id": doc_id})
        return 0

    def count(self) -> int:
        return -1  # not exposed by the LangChain wrapper

    def all_chunks(self) -> list[RetrievedChunk]:
        raise NotImplementedError(
            "PGVector does not expose a full dump; build BM25 from the ingest manifest instead."
        )


class PineconeStore(VectorStore):
    """Pinecone serverless."""

    def __init__(self, settings: Settings | None = None) -> None:
        from pinecone import Pinecone, ServerlessSpec

        self.settings = settings or get_settings()
        if not self.settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for the pinecone backend.")
        pc = Pinecone(api_key=self.settings.pinecone_api_key)
        existing = {i["name"] for i in pc.list_indexes()}
        if self.settings.pinecone_index not in existing:
            pc.create_index(
                name=self.settings.pinecone_index,
                dimension=self.settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = pc.Index(self.settings.pinecone_index)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        payload = [
            {
                "id": c.chunk_id,
                "values": list(v),
                # Pinecone has no document store, so the text rides in metadata.
                "metadata": {**c.to_metadata(), "text": c.text[:38000]},
            }
            for c, v in zip(chunks, vectors)
        ]
        for start in range(0, len(payload), 100):
            self._index.upsert(vectors=payload[start : start + 100])
        return len(payload)

    def search(
        self, vector: Sequence[float], k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        response = self._index.query(
            vector=list(vector), top_k=k, include_metadata=True, filter=where or None
        )
        out: list[RetrievedChunk] = []
        for rank, match in enumerate(response.get("matches", [])):
            meta = dict(match.get("metadata") or {})
            out.append(
                RetrievedChunk(
                    chunk_id=match["id"],
                    doc_id=meta.get("doc_id", ""),
                    text=meta.pop("text", ""),
                    metadata=meta,
                    dense_score=float(match.get("score", 0.0)),
                    dense_rank=rank + 1,
                    retriever="dense",
                )
            )
        return out

    def delete_document(self, doc_id: str) -> int:
        self._index.delete(filter={"doc_id": doc_id})
        return 0

    def count(self) -> int:
        return int(self._index.describe_index_stats().get("total_vector_count", 0))

    def all_chunks(self) -> list[RetrievedChunk]:
        raise NotImplementedError("Pinecone has no full-scan API; use the ingest manifest.")


class WeaviateStore(VectorStore):
    """Weaviate - the one backend with hybrid search inside the engine."""

    def __init__(self, settings: Settings | None = None) -> None:
        import weaviate
        from weaviate.classes.init import Auth

        self.settings = settings or get_settings()
        auth = Auth.api_key(self.settings.weaviate_api_key) if self.settings.weaviate_api_key else None
        self._client = weaviate.connect_to_custom(
            http_host=self.settings.weaviate_url.replace("http://", "").split(":")[0],
            http_port=int(self.settings.weaviate_url.rsplit(":", 1)[-1]) if ":" in self.settings.weaviate_url.replace("http://", "") else 8080,
            http_secure=self.settings.weaviate_url.startswith("https"),
            grpc_host=self.settings.weaviate_url.replace("http://", "").split(":")[0],
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=auth,
        )
        self._name = self.settings.collection_name.replace("_", "").title()
        if not self._client.collections.exists(self._name):
            self._client.collections.create(name=self._name)
        self._collection = self._client.collections.get(self._name)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        with self._collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                batch.add_object(
                    properties={**chunk.to_metadata(), "text": chunk.text},
                    vector=list(vector),
                    uuid=_uuid_for(chunk.chunk_id),
                )
        return len(chunks)

    def search(
        self, vector: Sequence[float], k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        from weaviate.classes.query import MetadataQuery

        response = self._collection.query.near_vector(
            near_vector=list(vector), limit=k, return_metadata=MetadataQuery(distance=True)
        )
        out: list[RetrievedChunk] = []
        for rank, obj in enumerate(response.objects):
            props = dict(obj.properties)
            out.append(
                RetrievedChunk(
                    chunk_id=props.get("chunk_id", str(obj.uuid)),
                    doc_id=props.get("doc_id", ""),
                    text=props.pop("text", ""),
                    metadata=props,
                    dense_score=1.0 - float(obj.metadata.distance or 0.0),
                    dense_rank=rank + 1,
                    retriever="dense",
                )
            )
        return out

    def delete_document(self, doc_id: str) -> int:
        from weaviate.classes.query import Filter

        self._collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
        return 0

    def count(self) -> int:
        return self._collection.aggregate.over_all(total_count=True).total_count

    def all_chunks(self) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for obj in self._collection.iterator():
            props = dict(obj.properties)
            out.append(
                RetrievedChunk(
                    chunk_id=props.get("chunk_id", str(obj.uuid)),
                    doc_id=props.get("doc_id", ""),
                    text=props.pop("text", ""),
                    metadata=props,
                )
            )
        return out


def _uuid_for(chunk_id: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


_BACKENDS = {
    "chroma": ChromaStore,
    "pgvector": PGVectorStore,
    "pinecone": PineconeStore,
    "weaviate": WeaviateStore,
}

_store_singleton: VectorStore | None = None


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton
    settings = settings or get_settings()
    backend = _BACKENDS.get(settings.vector_backend)
    if backend is None:
        raise ValueError(
            f"Unknown VECTOR_BACKEND '{settings.vector_backend}'. Options: {', '.join(_BACKENDS)}"
        )
    logger.info("Vector backend: %s", settings.vector_backend)
    _store_singleton = backend(settings)
    return _store_singleton


def reset_vector_store() -> None:
    global _store_singleton
    _store_singleton = None
