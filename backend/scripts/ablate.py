"""Ablation sweep: measure what each retrieval component is actually worth.

Retrieval stacks accumulate components because each one sounds sensible - and
nobody measures whether it helps on *this* corpus. Hybrid search, MMR and
cross-encoder re-ranking are all standard advice; that is not evidence.

This runs the same golden-set sample through several configurations and prints
them side by side. It calls no LLM, so a full sweep is free.

    python -m scripts.ablate -n 60
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

CONFIGS: list[tuple[str, dict]] = [
    ("dense only",            {"hybrid_enabled": False, "rerank_enabled": False, "mmr_enabled": False}),
    ("dense + mmr",           {"hybrid_enabled": False, "rerank_enabled": False, "mmr_enabled": True}),
    ("hybrid",                {"hybrid_enabled": True,  "rerank_enabled": False, "mmr_enabled": False}),
    ("hybrid + mmr",          {"hybrid_enabled": True,  "rerank_enabled": False, "mmr_enabled": True}),
    ("dense + rerank",        {"hybrid_enabled": False, "rerank_enabled": True,  "mmr_enabled": False}),
    ("hybrid + rerank",       {"hybrid_enabled": True,  "rerank_enabled": True,  "mmr_enabled": False}),
    ("hybrid + mmr + rerank", {"hybrid_enabled": True,  "rerank_enabled": True,  "mmr_enabled": True}),
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    from app.evaluation.runner import get_runner
    from app.retrieval import retriever as retriever_module

    runner = get_runner()
    rows: list[tuple[str, dict]] = []

    for label, overrides in CONFIGS:
        for key, value in overrides.items():
            setattr(settings, key, value)
        # The retriever caches its re-ranker at construction, so it has to be
        # rebuilt for a config change to actually take effect.
        retriever_module.reset_retriever()

        # The BM25 index lives in a module-level singleton that starts empty in
        # a fresh process. Without this rebuild the sparse arm is silently
        # skipped and every "hybrid" row secretly measures dense-only - which
        # is exactly the false negative this sweep exists to avoid.
        retriever = retriever_module.get_retriever(settings)
        if settings.hybrid_enabled and retriever.bm25.size == 0:
            retriever.refresh_bm25()

        print(f"  running: {label:24s} ", end="", flush=True)
        payload = await runner.run(
            sample_size=args.sample_size,
            seed=args.seed,
            retrieval_only=True,
            judge=False,
            concurrency=2,
        )
        metrics = payload["metrics"]["retrieval"]
        rows.append((label, metrics))
        sparse = retriever.bm25.size if settings.hybrid_enabled else 0
        print(
            f"nDCG={metrics['ndcg']['mean']:.3f}  ({payload['elapsed_s']:.0f}s, "
            f"bm25={sparse:,})"
        )

    print(f"\n{'=' * 82}")
    print(f"  {'configuration':<24} {'hit_rate':>9} {'recall':>9} {'prec':>9} {'MRR':>9} {'nDCG':>9}")
    print(f"{'=' * 82}")
    baseline = rows[0][1]
    for label, metrics in rows:
        delta = metrics["ndcg"]["mean"] - baseline["ndcg"]["mean"]
        marker = "" if label == rows[0][0] else f"  ({delta:+.3f})"
        print(
            f"  {label:<24} {metrics['hit_rate']['mean']:>9.3f} {metrics['span_recall']['mean']:>9.3f} "
            f"{metrics['precision']['mean']:>9.3f} {metrics['mrr']['mean']:>9.3f} "
            f"{metrics['ndcg']['mean']:>9.3f}{marker}"
        )
    print(f"{'=' * 82}\n")
    best = max(rows, key=lambda r: r[1]["ndcg"]["mean"])
    print(f"  Best by nDCG: {best[0]} ({best[1]['ndcg']['mean']:.3f})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
