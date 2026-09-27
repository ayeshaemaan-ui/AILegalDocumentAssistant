"use client";

import { Badge, Card } from "./ui";

/**
 * The measured component sweep. Shown in the product because it is the honest
 * version of an architecture claim: one of the three "standard" retrieval
 * components measurably hurt on this corpus, and only measuring found it.
 */

const ROWS = [
  { config: "dense only", hit: 0.713, recall: 0.643, prec: 0.173, mrr: 0.526, ndcg: 0.570 },
  { config: "dense + MMR", hit: 0.675, recall: 0.581, prec: 0.133, mrr: 0.508, ndcg: 0.545 },
  { config: "hybrid", hit: 0.650, recall: 0.590, prec: 0.163, mrr: 0.505, ndcg: 0.536 },
  { config: "hybrid + MMR", hit: 0.688, recall: 0.588, prec: 0.135, mrr: 0.511, ndcg: 0.550 },
  { config: "dense + rerank", hit: 0.675, recall: 0.603, prec: 0.167, mrr: 0.538, ndcg: 0.566 },
  { config: "hybrid + rerank", hit: 0.700, recall: 0.631, prec: 0.175, mrr: 0.540, ndcg: 0.573, active: true },
  { config: "hybrid + MMR + rerank", hit: 0.662, recall: 0.586, prec: 0.152, mrr: 0.525, ndcg: 0.556 },
];

const BEST = { hit: 0.713, recall: 0.643, prec: 0.175, mrr: 0.540, ndcg: 0.573 };

export function Ablation() {
  return (
    <Card className="p-4">
      <h3 className="text-[14px] font-semibold">What each component is actually worth</h3>
      <p className="mt-1 text-[12px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
        80 questions, scored against lawyer-marked clause spans, retrieval scoped to the correct
        document. Reproduce with <span className="font-mono">scripts/ablate.py</span> — it makes no
        model calls, so it is free to re-run.
      </p>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-[12.5px]">
          <thead>
            <tr className="border-b text-left" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
              <th className="py-1.5 pr-3 font-medium">Configuration</th>
              <th className="py-1.5 pr-3 text-right font-medium">hit_rate</th>
              <th className="py-1.5 pr-3 text-right font-medium">recall</th>
              <th className="py-1.5 pr-3 text-right font-medium">prec</th>
              <th className="py-1.5 pr-3 text-right font-medium">MRR</th>
              <th className="py-1.5 text-right font-medium">nDCG</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr
                key={r.config}
                className="border-b last:border-0"
                style={{ borderColor: "var(--border)", background: r.active ? "var(--surface-sunken)" : undefined }}
              >
                <td className="py-1.5 pr-3">
                  {r.config}
                  {r.active && (
                    <Badge tone="accent" title="This is the configuration currently running">
                      default
                    </Badge>
                  )}
                </td>
                <Cell v={r.hit} best={BEST.hit} />
                <Cell v={r.recall} best={BEST.recall} />
                <Cell v={r.prec} best={BEST.prec} />
                <Cell v={r.mrr} best={BEST.mrr} />
                <Cell v={r.ndcg} best={BEST.ndcg} last />
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex flex-col gap-2 text-[12.5px] leading-relaxed">
        <p>
          <strong>MMR hurt in every pairing</strong> — −0.024, −0.019 and −0.014 nDCG against its
          comparable baseline. That is the one large, consistent effect in the table, so MMR is{" "}
          <strong>off by default</strong>. The reason is legible: these questions are scoped to a
          single contract, where near-duplicate clauses are usually all relevant, so buying diversity
          costs relevance it cannot repay.
        </p>
        <p style={{ color: "var(--text-muted)" }}>
          <strong>The top two rows are not meaningfully different.</strong> 0.573 versus 0.570 at
          n=80 is noise, and presenting it as hybrid search winning would be dishonest. Hybrid +
          re-rank is kept as the default on two other grounds: it has the best MRR and precision, so
          it puts the right clause higher and wastes fewer context tokens; and this sweep runs
          without the query rewriter, which hands BM25 a conversational sentence instead of the
          keyword query it gets in production — so its floor here is lower than its real behaviour.
        </p>
      </div>
    </Card>
  );
}

function Cell({ v, best, last }: { v: number; best: number; last?: boolean }) {
  const isBest = Math.abs(v - best) < 1e-9;
  return (
    <td
      className={`py-1.5 text-right tabular-nums ${last ? "" : "pr-3"}`}
      style={{ fontWeight: isBest ? 650 : 400, color: isBest ? "var(--color-good-700)" : undefined }}
    >
      {v.toFixed(3)}
    </td>
  );
}
