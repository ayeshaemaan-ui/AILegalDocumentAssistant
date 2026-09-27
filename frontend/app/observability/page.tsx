"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Badge, Card, EmptyState, Spinner, Stat } from "@/components/ui";
import { TechStrip } from "@/components/TechStrip";

export default function ObservabilityPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [latency, setLatency] = useState<any>(null);
  const [cost, setCost] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [selected, setSelected] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [traceData, costData, healthData] = await Promise.all([
        api.traces(50),
        api.costSummary(7).catch(() => null),
        api.health().catch(() => null),
      ]);
      setTraces(traceData.traces ?? []);
      setLatency(traceData.latency ?? null);
      setCost(costData);
      setHealth(healthData);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 15000);
    return () => clearInterval(timer);
  }, [refresh]);

  const stages = cost?.by_stage ?? {};
  const maxStageCost = Math.max(1e-9, ...Object.values(stages).map((s: any) => s.cost_usd ?? 0));

  return (
    <div className="mx-auto max-w-[1200px] px-5 py-7">
      <header className="mb-5">
        <h1 className="text-[20px] font-semibold tracking-tight">Observability</h1>
        <p className="mt-1 max-w-3xl text-[13.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
          Every stage of every request opens a span. When an answer is wrong, this is what tells you
          whether retrieval missed the clause or generation ignored it — different failures with
          different fixes.
        </p>
        <TechStrip
          className="mt-2"
          items={["OpenTelemetry", "LangSmith", "Helicone", "SQLite span store", "per-stage cost ledger"]}
        />
      </header>

      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat
          label="Status"
          value={health?.status ?? "—"}
          tone={health?.status === "healthy" ? "good" : health?.status ? "warn" : undefined}
          hint={health ? `up ${Math.round((health.uptime_s ?? 0) / 60)}m` : undefined}
        />
        <Stat label="p50 latency" value={latency ? `${(latency.p50 / 1000).toFixed(2)}s` : "—"} hint={`${latency?.count ?? 0} requests / 24h`} />
        <Stat label="p95 latency" value={latency ? `${(latency.p95 / 1000).toFixed(2)}s` : "—"} />
        <Stat
          label="Spend today"
          value={`$${Number(cost?.spent_today_usd ?? 0).toFixed(4)}`}
          hint={`of $${Number(cost?.daily_ceiling_usd ?? 0).toFixed(2)} ceiling`}
          tone={
            cost && cost.daily_ceiling_usd > 0 && cost.spent_today_usd / cost.daily_ceiling_usd > 0.9
              ? "bad"
              : undefined
          }
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4">
          <h2 className="mb-1 text-[14px] font-semibold">Cost by pipeline stage</h2>
          <p className="mb-3 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
            Routing, rewriting and grading run on a small model and cost pennies. Generation is the bill.
          </p>
          {Object.keys(stages).length === 0 ? (
            <p className="text-[13px]" style={{ color: "var(--text-muted)" }}>
              No model calls recorded yet.
            </p>
          ) : (
            <div className="flex flex-col gap-2">
              {Object.entries(stages).map(([stage, data]: [string, any]) => (
                <div key={stage} className="flex items-center gap-2.5 text-[12.5px]">
                  <span className="w-40 truncate font-mono">{stage}</span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: "var(--surface-sunken)" }}>
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${Math.max(2, (data.cost_usd / maxStageCost) * 100)}%`, background: "var(--color-accent-500)" }}
                    />
                  </div>
                  <span className="w-20 text-right font-mono tabular-nums">${data.cost_usd.toFixed(5)}</span>
                  <span className="w-14 text-right tabular-nums" style={{ color: "var(--text-muted)" }}>
                    {data.calls}×
                  </span>
                </div>
              ))}
              <div className="mt-1.5 border-t pt-2 text-[12px]" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                Prompt-cache hit rate:{" "}
                <span className="font-mono font-medium" style={{ color: "var(--text)" }}>
                  {((cost?.cache_hit_rate ?? 0) * 100).toFixed(1)}%
                </span>{" "}
                — cached input bills at roughly a tenth of the normal rate.
              </div>
            </div>
          )}
        </Card>

        <Card className="p-4">
          <h2 className="mb-1 text-[14px] font-semibold">System checks</h2>
          <p className="mb-3 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
            Live readiness of each subsystem.
          </p>
          {health?.checks ? (
            <div className="flex flex-col gap-1.5 text-[12.5px]">
              {Object.entries(health.checks).map(([name, data]: [string, any]) => (
                <div key={name} className="flex items-start gap-2">
                  <Badge tone={data.ok === false ? "bad" : "good"}>{data.ok === false ? "fail" : "ok"}</Badge>
                  <span className="font-mono">{name}</span>
                  <span className="ml-auto text-right" style={{ color: "var(--text-muted)" }}>
                    {data.error ??
                      Object.entries(data)
                        .filter(([k]) => k !== "ok")
                        .map(([k, v]) => `${k}=${v}`)
                        .join(" · ")}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[13px]" style={{ color: "var(--text-muted)" }}>
              Unavailable.
            </p>
          )}
        </Card>
      </div>

      <Card className="mt-4 overflow-hidden">
        <div className="flex items-center gap-2 border-b px-4 py-2.5" style={{ borderColor: "var(--border)" }}>
          <h2 className="text-[14px] font-semibold">Recent traces</h2>
          {loading && <Spinner />}
          <span className="ml-auto text-[11.5px]" style={{ color: "var(--text-muted)" }}>
            auto-refreshes every 15s
          </span>
        </div>

        {!traces.length ? (
          <EmptyState title="No traces yet" body="Ask a question on the Research page and its full execution trace will appear here." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12.5px]">
              <thead>
                <tr className="border-b text-left" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                  <th className="px-4 py-2 font-medium">Trace</th>
                  <th className="px-4 py-2 font-medium">Operation</th>
                  <th className="px-4 py-2 font-medium">Question</th>
                  <th className="px-4 py-2 text-right font-medium">Duration</th>
                  <th className="px-4 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {traces.map((trace) => (
                  <tr
                    key={trace.trace_id}
                    onClick={async () => setSelected(await api.trace(trace.trace_id))}
                    className="cursor-pointer border-b last:border-0"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <td className="px-4 py-2 font-mono">{trace.trace_id?.slice(0, 8)}</td>
                    <td className="px-4 py-2 font-mono">{trace.name}</td>
                    <td className="max-w-[360px] px-4 py-2">
                      <span className="block truncate" style={{ color: "var(--text-muted)" }}>
                        {trace.attributes?.question ?? "—"}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {trace.duration_ms ? `${(trace.duration_ms / 1000).toFixed(2)}s` : "—"}
                    </td>
                    <td className="px-4 py-2">
                      <Badge tone={trace.status === "error" ? "bad" : "good"}>{trace.status}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {selected && (
        <Card className="mt-4 p-4">
          <div className="mb-3 flex items-center gap-2">
            <h2 className="text-[14px] font-semibold">Trace {selected.trace_id?.slice(0, 8)}</h2>
            <button onClick={() => setSelected(null)} className="ml-auto text-[12px] underline underline-offset-2" style={{ color: "var(--text-muted)" }}>
              Close
            </button>
          </div>
          <div className="flex flex-col gap-1.5">
            {selected.spans?.map((span: any) => (
              <div key={span.span_id} className="rounded-lg border p-2.5 text-[12px]" style={{ borderColor: "var(--border)" }}>
                <div className="flex items-center gap-2">
                  <span className="font-mono font-medium">{span.name}</span>
                  <Badge tone={span.status === "error" ? "bad" : "neutral"}>{span.status}</Badge>
                  <span className="ml-auto tabular-nums" style={{ color: "var(--text-muted)" }}>
                    {span.duration_ms?.toFixed(0)}ms
                  </span>
                </div>
                {span.attributes && Object.keys(span.attributes).length > 0 && (
                  <pre className="mt-1.5 overflow-x-auto whitespace-pre-wrap font-mono text-[11px]" style={{ color: "var(--text-muted)" }}>
                    {JSON.stringify(span.attributes, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
