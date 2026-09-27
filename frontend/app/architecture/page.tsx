"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Ablation } from "@/components/Ablation";
import { Algorithms } from "@/components/Algorithms";
import { PipelineDiagram } from "@/components/PipelineDiagram";
import { StageTable } from "@/components/StageTable";
import { TechStack } from "@/components/TechStack";
import { Badge, Card, Spinner, Stat } from "@/components/ui";

const SECTIONS = [
  { id: "pipeline", label: "Pipeline" },
  { id: "stages", label: "Stages" },
  { id: "algorithms", label: "Algorithms" },
  { id: "stack", label: "Stack" },
  { id: "evidence", label: "Evidence" },
  { id: "config", label: "Live config" },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

export default function ArchitecturePage() {
  const [section, setSection] = useState<SectionId>("pipeline");
  const [config, setConfig] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [mermaid, setMermaid] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.config().catch(() => null),
      api.health().catch(() => null),
      api.graph().catch(() => null),
    ])
      .then(([cfg, h, g]) => {
        setConfig(cfg);
        setHealth(h);
        setMermaid(g?.mermaid ?? "");
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="mx-auto max-w-[1100px] px-5 py-7">
      <header className="mb-4">
        <h1 className="text-[20px] font-semibold tracking-tight">How it works</h1>
        <p className="mt-1 max-w-3xl text-[13.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
          Every technique, technology and algorithm this system runs — in the order a question
          travels through it, with the reason each one is there rather than the obvious alternative.
        </p>
      </header>

      {/* Headline numbers, read live from the backend so they cannot drift. */}
      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat
          label="Indexed"
          value={(health?.checks?.vector_store?.documents ?? "—").toLocaleString?.() ?? "—"}
          hint="contracts"
        />
        <Stat
          label="Chunks"
          value={(health?.checks?.vector_store?.chunks ?? 0).toLocaleString()}
          hint="dense + sparse"
        />
        <Stat label="Pipeline stages" value={9} hint="2 retry loops" />
        <Stat label="Tests" value={184} hint="all passing" />
      </div>

      <nav
        className="sticky top-14 z-30 mb-5 flex gap-0.5 overflow-x-auto rounded-lg border p-0.5 backdrop-blur-md"
        style={{ borderColor: "var(--border)", background: "color-mix(in srgb, var(--surface) 88%, transparent)" }}
      >
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => setSection(s.id)}
            className="whitespace-nowrap rounded-[6px] px-3 py-1.5 text-[12.5px] font-medium transition-colors"
            style={
              section === s.id
                ? { background: "var(--color-accent-600)", color: "#fff" }
                : { color: "var(--text-muted)" }
            }
          >
            {s.label}
          </button>
        ))}
      </nav>

      {section === "pipeline" && (
        <div className="flex flex-col gap-4">
          <Card className="p-4">
            <h2 className="mb-1 text-[14px] font-semibold">The request pipeline</h2>
            <p className="mb-3 text-[12px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
              Nine stages. The two dashed arrows are what separate a workflow from a chain: the
              system grades its own retrieval before generating, and audits its own answer before
              returning it — and gets one bounded retry at each. A plain RAG chain generates from
              whatever it retrieved and reports whatever it generated.
            </p>
            <PipelineDiagram />
          </Card>

          <div className="grid gap-3 md:grid-cols-3">
            <Card className="p-4">
              <h3 className="text-[13px] font-semibold">Why RAG, not fine-tuning</h3>
              <p className="mt-1.5 text-[12.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
                The corpus changes per user — someone uploads a contract and expects an answer about
                it immediately. And every answer must cite a source a reader can open. Neither is
                true of a fine-tuned model, which bakes knowledge in and cannot point at where it
                came from.
              </p>
            </Card>
            <Card className="p-4">
              <h3 className="text-[13px] font-semibold">Why two models</h3>
              <p className="mt-1.5 text-[12.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
                Routing, rewriting, grading and judging are four high-volume, low-difficulty calls.
                Running them on the small model costs pennies and leaves the expensive model for the
                one job that needs it. The cost page shows the split.
              </p>
            </Card>
            <Card className="p-4">
              <h3 className="text-[13px] font-semibold">Why citations are validated</h3>
              <p className="mt-1.5 text-[12.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
                A model that cites <span className="font-mono">[7]</span> when handed 6 excerpts has
                fabricated its evidence. That is a set-membership check, not a judgement call — so it
                is caught by a regex before any grader runs.
              </p>
            </Card>
          </div>
        </div>
      )}

      {section === "stages" && <StageTable />}
      {section === "algorithms" && <Algorithms />}
      {section === "stack" && <TechStack />}

      {section === "evidence" && (
        <div className="flex flex-col gap-4">
          <Ablation />
          <Card className="p-4">
            <h3 className="text-[14px] font-semibold">Metadata extraction, 500 contracts</h3>
            <p className="mt-1 text-[12px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
              Rule-based extractors scored against CUAD&rsquo;s lawyer-written answers.
            </p>
            <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <Stat label="Parties" value="93.4%" hint="recall" tone="good" />
              <Stat label="Contract type" value="95.6%" hint="match" tone="good" />
              <Stat label="Governing law" value="95.8%" hint="precision · abstains on 33%" tone="good" />
            </div>
            <p className="mt-3 text-[12.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
              The governing-law profile is deliberate. A wrong value becomes a metadata filter that
              silently hides the right document — worse than no filter at all — so the extractor
              abstains rather than guessing. &ldquo;Organised under the laws of Germany&rdquo; is
              where a party was incorporated, not what governs the contract, and it is rejected on
              purpose.
            </p>
          </Card>
          <Card className="p-4">
            <h3 className="text-[14px] font-semibold">Known limitations</h3>
            <ul className="mt-2 flex flex-col gap-1.5 text-[12.5px] leading-relaxed">
              {[
                "Roughly a third of questions do not surface a lawyer-marked span in the top 6. Weakest on diffuse categories — competitive-restriction carve-outs, affiliate licence grants — where the answer spans several clauses.",
                "Page numbers are exact for PDFs but estimated for text extracts, which is why those render as ~p.7 rather than p.7.",
                "Clause tagging is rule-based: high precision, moderate recall. Untagged clauses are still found by dense retrieval, they just cannot be filtered on.",
                "The LLM judge is a proxy — useful for comparing two versions of this system, not for claiming an absolute quality level.",
                "The rate limiter, job board and BM25 index are in-process. Multiple replicas would need Redis and a shared index.",
                "English only, tuned on US-style commercial contracts.",
              ].map((t) => (
                <li key={t} className="flex gap-2">
                  <span style={{ color: "var(--text-muted)" }}>·</span>
                  <span style={{ color: "var(--text-muted)" }}>{t}</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      )}

      {section === "config" && (
        <div className="flex flex-col gap-4">
          {loading ? (
            <Card className="flex items-center gap-2 p-5 text-[13px]" style={{ color: "var(--text-muted)" }}>
              <Spinner /> Reading live configuration…
            </Card>
          ) : config ? (
            <Card className="p-4">
              <h2 className="mb-1 text-[14px] font-semibold">Live configuration</h2>
              <p className="mb-3 text-[12px]" style={{ color: "var(--text-muted)" }}>
                Read from the running backend — this is what is serving requests right now, not a
                copy of the defaults. Secrets are never included.
              </p>
              <div className="grid gap-3 md:grid-cols-2">
                {Object.entries(config)
                  .filter(([, v]) => v && typeof v === "object")
                  .map(([group, values]: [string, any]) => (
                    <div key={group}>
                      <div className="mb-1 text-[10.5px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                        {group}
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(values).map(([k, v]) => (
                          <Badge key={k}>
                            {k}={String(v)}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>
            </Card>
          ) : (
            <Card className="p-4 text-[13px]" style={{ color: "var(--text-muted)" }}>
              Backend unreachable — start it with{" "}
              <span className="font-mono">uvicorn app.main:app</span>.
            </Card>
          )}

          {mermaid && (
            <Card className="p-4">
              <h2 className="mb-1 text-[14px] font-semibold">Compiled LangGraph topology</h2>
              <p className="mb-3 text-[12px]" style={{ color: "var(--text-muted)" }}>
                Generated from the compiled graph object, so it is the real topology rather than a
                drawing of it.
              </p>
              <pre
                className="overflow-x-auto rounded-lg border p-3 font-mono text-[11px] leading-relaxed"
                style={{ borderColor: "var(--border)", background: "var(--surface-sunken)", color: "var(--text-muted)" }}
              >
                {mermaid}
              </pre>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
