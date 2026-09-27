"use client";

import type { AskResponse } from "@/lib/types";
import { Badge, Disclosure } from "./ui";

/**
 * Exposes what the pipeline actually did for this question: how it was routed,
 * how the query was rewritten, how many candidates each retrieval arm produced,
 * what survived fusion and re-ranking, and what it cost.
 *
 * This is deliberately visible rather than hidden in logs. When an answer looks
 * wrong, this panel is what tells you whether retrieval missed the clause or
 * generation ignored it - and those have completely different fixes.
 */
export function PipelineTrace({ meta }: { meta: AskResponse }) {
  const r = meta.retrieval ?? {};
  const timings = r.timings_ms ?? {};
  const usage = meta.usage ?? {};

  const stages: Array<[string, string | number | undefined]> = [
    ["route", meta.intent ?? "—"],
    ["dense", r.dense_count],
    ["sparse", r.sparse_count],
    ["fused", r.fused_count],
    ["after MMR", r.after_mmr],
    ["final", r.final_count],
  ];

  return (
    <Disclosure
      label="Pipeline"
      right={
        <span className="flex items-center gap-1.5">
          <Badge>{meta.elapsed_ms.toFixed(0)}ms</Badge>
          {typeof usage.cost_usd === "number" && usage.cost_usd > 0 && (
            <Badge>${usage.cost_usd.toFixed(4)}</Badge>
          )}
        </span>
      }
    >
      <div className="rounded-lg border p-2.5 text-[11.5px]" style={{ borderColor: "var(--border)", background: "var(--surface-sunken)" }}>
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
          {stages.map(([label, value], i) => (
            <span key={label} className="flex items-center gap-1.5">
              {i > 0 && <span style={{ color: "var(--text-muted)" }}>→</span>}
              <span style={{ color: "var(--text-muted)" }}>{label}</span>
              <span className="font-mono font-medium tabular-nums">{value ?? "—"}</span>
            </span>
          ))}
          {r.reranked && <Badge tone="accent">re-ranked</Badge>}
          {r.fallback_used && <Badge tone="warn">filter relaxed</Badge>}
        </div>

        {meta.search_query && meta.search_query !== "" && (
          <div className="mt-2 border-t pt-2" style={{ borderColor: "var(--border)" }}>
            <span style={{ color: "var(--text-muted)" }}>Rewritten query: </span>
            <span className="font-mono">{meta.search_query}</span>
          </div>
        )}

        {meta.clause_types?.length > 0 && (
          <div className="mt-1.5 flex flex-wrap items-center gap-1">
            <span style={{ color: "var(--text-muted)" }}>Clause filter:</span>
            {meta.clause_types.map((s) => (
              <Badge key={s} tone="accent">{s.replace(/_/g, " ")}</Badge>
            ))}
          </div>
        )}

        {Object.keys(timings).length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 font-mono" style={{ color: "var(--text-muted)" }}>
            {Object.entries(timings).map(([stage, ms]) => (
              <span key={stage}>
                {stage} {Number(ms).toFixed(0)}ms
              </span>
            ))}
          </div>
        )}

        {meta.route_reason && (
          <div className="mt-1.5 italic" style={{ color: "var(--text-muted)" }}>
            {meta.route_reason}
          </div>
        )}

        {typeof usage.total_tokens === "number" && usage.total_tokens > 0 && (
          <div className="mt-1.5 border-t pt-1.5 font-mono" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
            {usage.calls} model call{usage.calls === 1 ? "" : "s"} · {usage.total_tokens.toLocaleString()} tokens
          </div>
        )}
      </div>
    </Disclosure>
  );
}
