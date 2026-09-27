"""Index the CUAD corpus.

    python -m scripts.ingest_cuad --limit 50                  # subset, fast
    python -m scripts.ingest_cuad --limit 510                 # full corpus
    python -m scripts.ingest_cuad --strategy recursive --reset  # A/B a strategy

`--reset` wipes the collection first, which is what you want when comparing
chunking strategies: leaving the old chunks in place would mix two strategies
in one index and make the evaluation meaningless.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index CUAD contracts into the vector store.")
    parser.add_argument("--limit", type=int, default=50, help="How many contracts to index.")
    parser.add_argument("--strategy", choices=["recursive", "legal_structural", "semantic"])
    parser.add_argument("--source", choices=["txt", "pdf"], default="txt")
    parser.add_argument("--reset", action="store_true", help="Clear the collection first.")
    parser.add_argument("--force", action="store_true", help="Re-index documents already present.")
    args = parser.parse_args()

    settings = get_settings()
    if args.strategy:
        settings.chunk_strategy = args.strategy

    directory = settings.cuad_dir / ("full_contract_txt" if args.source == "txt" else "full_contract_pdf")
    if not directory.exists():
        print(f"CUAD not found at {directory}", file=sys.stderr)
        print("Set CUAD_DIR in .env to the folder containing CUAD_v1.", file=sys.stderr)
        return 1

    if args.reset:
        from app.retrieval.vectorstore import get_vector_store

        store = get_vector_store(settings)
        if hasattr(store, "reset"):
            store.reset()
            print("Collection reset.")

    from app.ingestion.pipeline import get_pipeline

    pipeline = get_pipeline(settings)
    print(
        f"Indexing up to {args.limit} contracts from {directory.name}\n"
        f"  strategy   {settings.chunk_strategy}\n"
        f"  embeddings {settings.embedding_model}\n"
        f"  store      {settings.vector_backend}\n"
    )

    started = time.perf_counter()
    totals = {"chunks": 0, "indexed": 0, "skipped": 0, "failed": 0}

    def progress(index: int, total: int, result) -> None:
        if result.error:
            totals["failed"] += 1
            status = f"FAILED  {result.error[:50]}"
        elif result.skipped:
            totals["skipped"] += 1
            status = "skipped (already indexed)"
        else:
            totals["indexed"] += 1
            totals["chunks"] += result.chunk_count
            status = f"{result.chunk_count:4d} chunks  {result.elapsed_s:5.1f}s"
        print(f"  [{index:3d}/{total}] {result.title[:58]:60s} {status}", flush=True)

    pipeline.ingest_directory(
        directory,
        limit=args.limit,
        patterns=("*.txt",) if args.source == "txt" else ("*.pdf",),
        strategy=args.strategy,
        force=args.force,
        progress=progress,
    )

    # BM25 is an in-memory index built from the store, so it must be rebuilt
    # after ingestion or hybrid search would run with a stale sparse arm.
    from app.retrieval.retriever import get_retriever

    bm25_chunks = get_retriever(settings).refresh_bm25()
    elapsed = time.perf_counter() - started

    print(
        f"\nDone in {elapsed:.1f}s\n"
        f"  indexed  {totals['indexed']} documents, {totals['chunks']:,} chunks\n"
        f"  skipped  {totals['skipped']}   failed {totals['failed']}\n"
        f"  BM25     {bm25_chunks:,} chunks\n"
        f"  store    {pipeline.stats()['chunks']:,} vectors total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
