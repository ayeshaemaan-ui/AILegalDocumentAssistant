"""Run an evaluation from the command line.

    python -m scripts.run_eval --retrieval-only -n 80      # free, no API key
    python -m scripts.run_eval -n 25                       # full, costs money
    python -m scripts.run_eval --compare RUN_A RUN_B       # regression check

`--retrieval-only` is the mode to live in while tuning. It scores retrieval
against CUAD's lawyer-marked spans with no model calls at all, so you can sweep
chunk sizes or hybrid weights for free and know the number moved because the
retriever changed, not because the generator was having a good day.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.runner import get_runner  # noqa: E402


def print_report(payload: dict) -> None:
    metrics = payload["metrics"]
    config = payload["config"]

    print(f"\n{'=' * 74}")
    print(f"  Run {payload['run_id']}   mode={payload['mode']}   n={payload['sample_size']}")
    print(f"{'=' * 74}")
    print(
        f"  chunking={config['chunk_strategy']}({config['chunk_size']}/{config['chunk_overlap']})  "
        f"hybrid={config['hybrid_enabled']}  rerank={config['rerank_backend'] if config['rerank_enabled'] else 'off'}  "
        f"mmr={config['mmr_enabled']}  k={config['retrieval_top_k']}->{config['final_top_k']}"
    )
    print(f"  elapsed={payload['elapsed_s']}s  cost=${payload['cost_usd']:.4f}\n")

    print("  RETRIEVAL (vs lawyer-annotated spans)")
    print(f"    {'metric':<12} {'mean':>8} {'p10':>8} {'p50':>8} {'p90':>8}")
    for name, values in metrics.get("retrieval", {}).items():
        print(
            f"    {name:<12} {values['mean']:>8.3f} {values['p10']:>8.3f} "
            f"{values['p50']:>8.3f} {values['p90']:>8.3f}"
        )

    latency = metrics.get("latency_ms", {})
    if latency:
        print(f"\n  LATENCY   mean={latency.get('mean', 0):.0f}ms  p90={latency.get('p90', 0):.0f}ms")

    if quality := metrics.get("answer_quality"):
        print("\n  ANSWER QUALITY (LLM judge vs lawyer reference, 1-5)")
        for key in ("correctness", "completeness", "citation_quality", "overall"):
            if key in quality:
                print(f"    {key:<18} {quality[key]:.2f}")

    if faithfulness := metrics.get("faithfulness"):
        print("\n  FAITHFULNESS")
        print(f"    hallucination_rate   {faithfulness.get('hallucination_rate', 0):.3f}")
        print(f"    mean_grounded_ratio  {faithfulness.get('mean_grounded_ratio', 0):.3f}")
        print(f"    verdicts             {faithfulness.get('verdicts', {})}")

    if citations := metrics.get("citations"):
        print("\n  CITATIONS")
        for key, value in citations.items():
            print(f"    {key:<32} {value}")

    if weakest := metrics.get("weakest_categories"):
        print("\n  WEAKEST CATEGORIES (by nDCG) - where retrieval needs work")
        for entry in weakest:
            print(f"    {entry['category'][:40]:<42} nDCG={entry['ndcg']:.3f}  n={entry['n']}")

    errors = metrics.get("errors", 0)
    if errors:
        print(f"\n  {errors} example(s) errored")
    print()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--list", action="store_true", help="List previous runs.")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a report.")
    args = parser.parse_args()

    runner = get_runner()

    if args.list:
        for run in runner.list_runs(20):
            ndcg = run["metrics"].get("retrieval", {}).get("ndcg", {}).get("mean", 0)
            print(
                f"  {run['run_id']}  {run['mode']:<14} n={run['sample_size']:<4} "
                f"nDCG={ndcg:.3f}  ${run['cost_usd']:.4f}  "
                f"{run['config'].get('chunk_strategy')}/{run['config'].get('rerank_backend')}"
            )
        return 0

    if args.compare:
        result = runner.compare_runs(*args.compare)
        print(json.dumps(result, indent=2))
        return 0

    def progress(done: int, total: int, result) -> None:
        bar = "#" * int(24 * done / total)
        print(f"\r  [{bar:<24}] {done}/{total}", end="", flush=True)

    payload = await runner.run(
        sample_size=args.sample_size,
        seed=args.seed,
        categories=args.categories,
        retrieval_only=args.retrieval_only,
        judge=not args.no_judge,
        concurrency=args.concurrency,
        progress=progress,
    )
    print()
    if args.json:
        payload.pop("examples", None)
        print(json.dumps(payload, indent=2))
    else:
        print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
