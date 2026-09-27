"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { EvalRun } from "@/lib/types";
import { Badge, Button, Card, EmptyState, Spinner, Stat } from "@/components/ui";
import { TechStrip } from "@/components/TechStrip";

const METRIC_HELP: Record<string, string> = {
  hit_rate: "Share of questions where at least one lawyer-marked clause made the top k. The floor — if this is low, the model was never shown the answer.",
  span_recall: "Share of the lawyer-marked passages that were retrieved. Measured in spans, not chunks.",
  precision: "Share of returned chunks that were relevant. Low precision means context tokens spent on noise.",
  mrr: "Mean reciprocal rank of the first relevant chunk. High hit_rate with low MRR means the right clause is being found but buried — a re-ranking problem.",
  ndcg: "Rank-weighted relevance. The best single-number summary.",
};

export default function EvaluationPage() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [golden, setGolden] = useState<any>(null);
  const [selected, setSelected] = useState<EvalRun | null>(null);
  const [job, setJob] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sampleSize, setSampleSize] = useState(40);
  const [retrievalOnly, setRetrievalOnly] = useState(true);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [runsData, goldenData] = await Promise.all([
        api.evalRuns(),
        api.goldenSet().catch(() => null),
      ]);
      setRuns(runsData.runs);
      setGolden(goldenData);
      setSelected((current) => current ?? runsData.runs[0] ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load evaluation runs.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, [refresh]);

  const pollJob = useCallback(
    async (jobId: string) => {
      try {
        const status = await api.evalJob(jobId);
        setJob(status);
        if (status.status === "running" || status.status === "queued") {
          pollTimer.current = setTimeout(() => pollJob(jobId), 1500);
        } else {
          await refresh();
          if (status.run_id) {
            const full = await api.evalRun(status.run_id);
            setSelected(full);
          }
        }
      } catch {
        setJob(null);
      }
    },
    [refresh],
  );

  async function startRun() {
    setError(null);
    try {
      const started = await api.startEval({
        sample_size: sampleSize,
        retrieval_only: retrievalOnly,
        judge: !retrievalOnly,
      });
      setJob(started);
      pollJob(started.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start the evaluation.");
    }
  }

  const running = job && (job.status === "running" || job.status === "queued");
  const metrics = selected?.metrics;
  const retrieval = metrics?.retrieval;

  return (
    <div className="mx-auto max-w-[1200px] px-5 py-7">
      <header className="mb-5">
        <h1 className="text-[20px] font-semibold tracking-tight">Evaluation</h1>
        <p className="mt-1 max-w-3xl text-[13.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
          Retrieval is scored against CUAD&rsquo;s lawyer-annotated clause spans, so these numbers are
          objective — no model judges them. Answer quality is graded separately by an LLM against the
          lawyer&rsquo;s reference answer, which is a proxy and is labelled as one.
        </p>
        <TechStrip
          className="mt-2"
          items={["CUAD ground truth", "hit_rate · MRR · nDCG", "span matching", "LLM-as-judge"]}
        />
      </header>

      {golden && (
        <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Golden examples" value={(golden.positive_examples ?? 0).toLocaleString()} hint="with marked spans" />
          <Stat label="Documents" value={golden.documents ?? 0} />
          <Stat label="Clause categories" value={Object.keys(golden.by_category ?? {}).length} />
          <Stat label="Runs recorded" value={runs.length} />
        </div>
      )}

      <Card className="mb-5 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
              Sample size
            </span>
            <input
              type="number"
              min={1}
              max={300}
              value={sampleSize}
              onChange={(e) => setSampleSize(Math.max(1, Math.min(300, Number(e.target.value))))}
              className="w-24 rounded-lg border px-2.5 py-1.5 text-[13px] outline-none"
              style={{ borderColor: "var(--border)", background: "var(--bg)", color: "var(--text)" }}
            />
          </label>

          <label className="flex cursor-pointer items-center gap-2 pb-1.5 text-[13px]">
            <input type="checkbox" checked={retrievalOnly} onChange={(e) => setRetrievalOnly(e.target.checked)} />
            <span>
              Retrieval only
              <span className="ml-1.5 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
                (free — no model calls)
              </span>
            </span>
          </label>

          <Button variant="primary" onClick={startRun} disabled={running}>
            {running ? "Running…" : "Run evaluation"}
          </Button>

          {!retrievalOnly && (
            <span className="pb-1.5 text-[11.5px]" style={{ color: "var(--color-warn-700)" }}>
              Full mode generates and grades every answer — this costs money.
            </span>
          )}
        </div>

        {job && (
          <div className="mt-3 rounded-lg border p-3" style={{ borderColor: "var(--border)", background: "var(--surface-sunken)" }}>
            <div className="flex items-center gap-2 text-[12.5px]">
              {running && <Spinner />}
              <span className="font-medium">
                {job.status === "completed"
                  ? `Run ${job.run_id} complete`
                  : job.status === "failed"
                    ? "Run failed"
                    : `Evaluating… ${job.processed ?? 0}/${job.total ?? "?"}`}
              </span>
              {job.error && <span style={{ color: "var(--color-bad-500)" }}>{job.error}</span>}
              {job.status === "completed" && (
                <span style={{ color: "var(--text-muted)" }}>
                  {job.elapsed_s}s · ${Number(job.cost_usd ?? 0).toFixed(4)}
                </span>
              )}
            </div>
            {running && job.total > 0 && (
              <div className="mt-2 h-1 overflow-hidden rounded-full" style={{ background: "var(--border)" }}>
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${Math.round(((job.processed ?? 0) / job.total) * 100)}%`, background: "var(--color-accent-500)" }}
                />
              </div>
            )}
          </div>
        )}

        {error && (
          <p className="mt-2.5 text-[12.5px]" style={{ color: "var(--color-bad-500)" }}>
            {error}
          </p>
        )}
      </Card>

      {loading ? (
        <Card className="flex items-center gap-2 p-5 text-[13px]" style={{ color: "var(--text-muted)" }}>
          <Spinner /> Loading…
        </Card>
      ) : !runs.length ? (
        <Card>
          <EmptyState
            title="No evaluation runs yet"
            body="Run a retrieval-only evaluation to score the pipeline against lawyer-annotated ground truth. It makes no model calls, so it is free."
          />
        </Card>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
          <Card className="overflow-hidden">
            <div className="border-b px-3.5 py-2 text-[11px] font-semibold uppercase tracking-wide" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
              Runs
            </div>
            <div className="max-h-[560px] overflow-y-auto">
              {runs.map((run) => {
                const active = selected?.run_id === run.run_id;
                const ndcg = run.metrics?.retrieval?.ndcg?.mean ?? 0;
                return (
                  <button
                    key={run.run_id}
                    onClick={async () => setSelected(await api.evalRun(run.run_id))}
                    className="flex w-full flex-col gap-0.5 border-b px-3.5 py-2.5 text-left last:border-0"
                    style={{ borderColor: "var(--border)", background: active ? "var(--surface-sunken)" : "transparent" }}
                  >
                    <span className="flex items-center gap-1.5">
                      <span className="font-mono text-[11.5px]">{run.run_id.slice(0, 8)}</span>
                      <Badge tone={run.mode === "full" ? "accent" : "neutral"}>{run.mode}</Badge>
                    </span>
                    <span className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>
                      nDCG {ndcg.toFixed(3)} · n={run.sample_size} ·{" "}
                      {new Date(run.created_at * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                    </span>
                  </button>
                );
              })}
            </div>
          </Card>

          <div className="flex flex-col gap-4">
            {retrieval && (
              <Card className="p-4">
                <h2 className="mb-3 text-[14px] font-semibold">Retrieval quality</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-[13px]">
                    <thead>
                      <tr className="border-b text-left" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                        <th className="py-1.5 pr-3 font-medium">Metric</th>
                        <th className="py-1.5 pr-3 text-right font-medium">Mean</th>
                        <th className="py-1.5 pr-3 text-right font-medium">p10</th>
                        <th className="py-1.5 pr-3 text-right font-medium">p50</th>
                        <th className="py-1.5 text-right font-medium">p90</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(retrieval).map(([name, values]) => (
                        <tr key={name} className="border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                          <td className="py-1.5 pr-3" title={METRIC_HELP[name]}>
                            <span className="font-mono">{name}</span>
                            {METRIC_HELP[name] && (
                              <span className="ml-1 cursor-help" style={{ color: "var(--text-muted)" }}>
                                ⓘ
                              </span>
                            )}
                          </td>
                          <td className="py-1.5 pr-3 text-right font-medium tabular-nums">{values.mean.toFixed(3)}</td>
                          <td className="py-1.5 pr-3 text-right tabular-nums" style={{ color: "var(--text-muted)" }}>{values.p10.toFixed(3)}</td>
                          <td className="py-1.5 pr-3 text-right tabular-nums" style={{ color: "var(--text-muted)" }}>{values.p50.toFixed(3)}</td>
                          <td className="py-1.5 text-right tabular-nums" style={{ color: "var(--text-muted)" }}>{values.p90.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}

            {metrics?.answer_quality && (
              <Card className="p-4">
                <h2 className="mb-1 text-[14px] font-semibold">Answer quality</h2>
                <p className="mb-3 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
                  Graded 1–5 by an LLM judge against the lawyer&rsquo;s reference answer. A proxy, useful for comparing runs.
                </p>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  {Object.entries(metrics.answer_quality)
                    .filter(([k]) => k !== "graded")
                    .map(([key, value]) => (
                      <Stat key={key} label={key.replace(/_/g, " ")} value={Number(value).toFixed(2)} hint="of 5" />
                    ))}
                </div>
              </Card>
            )}

            {metrics?.faithfulness && (
              <Card className="p-4">
                <h2 className="mb-3 text-[14px] font-semibold">Faithfulness</h2>
                <div className="grid grid-cols-2 gap-3">
                  <Stat
                    label="Hallucination rate"
                    value={`${((metrics.faithfulness.hallucination_rate ?? 0) * 100).toFixed(1)}%`}
                    tone={(metrics.faithfulness.hallucination_rate ?? 0) > 0.05 ? "bad" : "good"}
                  />
                  <Stat
                    label="Mean grounded ratio"
                    value={(metrics.faithfulness.mean_grounded_ratio ?? 0).toFixed(3)}
                    hint="claims matched to an excerpt"
                  />
                </div>
              </Card>
            )}

            {metrics?.weakest_categories && metrics.weakest_categories.length > 0 && (
              <Card className="p-4">
                <h2 className="mb-1 text-[14px] font-semibold">Weakest clause categories</h2>
                <p className="mb-3 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
                  Where retrieval is failing. These are the categories to target next.
                </p>
                <div className="flex flex-col gap-1.5">
                  {metrics.weakest_categories.map((entry) => (
                    <div key={entry.category} className="flex items-center gap-2.5 text-[12.5px]">
                      <span className="w-56 truncate">{entry.category}</span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: "var(--surface-sunken)" }}>
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.max(2, entry.ndcg * 100)}%`,
                            background: entry.ndcg < 0.3 ? "var(--color-bad-500)" : entry.ndcg < 0.6 ? "var(--color-warn-500)" : "var(--color-good-500)",
                          }}
                        />
                      </div>
                      <span className="w-12 text-right font-mono tabular-nums">{entry.ndcg.toFixed(3)}</span>
                      <span className="w-8 text-right" style={{ color: "var(--text-muted)" }}>n={entry.n}</span>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {selected?.config && (
              <Card className="p-4">
                <h2 className="mb-2 text-[14px] font-semibold">Configuration that produced this run</h2>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(selected.config).map(([key, value]) => (
                    <Badge key={key}>
                      {key}={String(value)}
                    </Badge>
                  ))}
                </div>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
