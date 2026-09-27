"use client";

/**
 * The request pipeline as a diagram.
 *
 * Inline SVG rather than a charting library: this is a fixed topology, not
 * data, and inline SVG inherits the theme tokens so it works in light and dark
 * without a second palette. Every box names the technology that runs it, and
 * the two dashed arrows are the self-correction loops — the thing that makes
 * this a workflow rather than a chain.
 */

type Stage = {
  id: string;
  label: string;
  tech: string;
  kind: "guard" | "llm" | "search" | "out";
};

const STAGES: Stage[] = [
  { id: "guard", label: "Guard input", tech: "regex rules", kind: "guard" },
  { id: "route", label: "Route intent", tech: "Haiku 4.5 · structured output", kind: "llm" },
  { id: "rewrite", label: "Rewrite query", tech: "Haiku 4.5 · one-shot", kind: "llm" },
  { id: "retrieve", label: "Retrieve", tech: "BGE + BM25 → RRF", kind: "search" },
  { id: "rerank", label: "Re-rank", tech: "MiniLM cross-encoder", kind: "search" },
  { id: "grade", label: "Grade context", tech: "Haiku 4.5 · schema", kind: "llm" },
  { id: "generate", label: "Generate", tech: "Opus 5 · streaming", kind: "llm" },
  { id: "verify", label: "Verify", tech: "regex + Haiku audit", kind: "guard" },
  { id: "final", label: "Answer + citations", tech: "SSE → React", kind: "out" },
];

const FILL: Record<Stage["kind"], string> = {
  guard: "var(--color-warn-100)",
  llm: "var(--color-accent-100)",
  search: "var(--color-good-100)",
  out: "var(--surface-sunken)",
};
const STROKE: Record<Stage["kind"], string> = {
  guard: "var(--color-warn-500)",
  llm: "var(--color-accent-500)",
  search: "var(--color-good-500)",
  out: "var(--color-ink-400)",
};
const TEXT: Record<Stage["kind"], string> = {
  guard: "var(--color-warn-700)",
  llm: "var(--color-accent-700)",
  search: "var(--color-good-700)",
  out: "var(--text)",
};

const BOX_W = 210;
const BOX_H = 52;
const GAP = 26;
const LEFT = 130;
const TOP = 16;
const LOOP_X = LEFT + BOX_W + 46;

export function PipelineDiagram() {
  const height = TOP + STAGES.length * (BOX_H + GAP);

  return (
    <div className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${LOOP_X + 130} ${height}`}
        width="100%"
        style={{ minWidth: 560, maxWidth: 720 }}
        role="img"
        aria-label="LegalLens request pipeline: guard, route, rewrite, retrieve, re-rank, grade, generate, verify, answer, with two self-correction loops"
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-ink-400)" />
          </marker>
          <marker id="arrow-loop" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-warn-500)" />
          </marker>
        </defs>

        {STAGES.map((stage, i) => {
          const y = TOP + i * (BOX_H + GAP);
          const next = STAGES[i + 1];
          return (
            <g key={stage.id}>
              <rect
                x={LEFT} y={y} width={BOX_W} height={BOX_H} rx={9}
                fill={FILL[stage.kind]} stroke={STROKE[stage.kind]} strokeWidth={1.2}
              />
              <text x={LEFT + 14} y={y + 21} fontSize={12.5} fontWeight={600} fill={TEXT[stage.kind]}>
                {stage.label}
              </text>
              <text x={LEFT + 14} y={y + 38} fontSize={10.5} fontFamily="var(--font-mono)" fill={TEXT[stage.kind]} opacity={0.75}>
                {stage.tech}
              </text>
              <text x={LEFT - 14} y={y + 31} fontSize={10.5} textAnchor="end" fontFamily="var(--font-mono)" fill="var(--text-muted)">
                {i + 1}
              </text>
              {next && (
                <line
                  x1={LEFT + BOX_W / 2} y1={y + BOX_H}
                  x2={LEFT + BOX_W / 2} y2={y + BOX_H + GAP}
                  stroke="var(--color-ink-400)" strokeWidth={1.2} markerEnd="url(#arrow)"
                />
              )}
            </g>
          );
        })}

        {/* Loop 1: grade (6) decides the context is too thin -> widen and retrieve again */}
        <LoopArrow fromIndex={5} toIndex={3} label="context too thin" />
        {/* Loop 2: verify (8) finds the answer ungrounded -> regenerate once */}
        <LoopArrow fromIndex={7} toIndex={6} label="not grounded" />
      </svg>

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]" style={{ color: "var(--text-muted)" }}>
        <Legend swatch="var(--color-warn-500)" label="deterministic guard" />
        <Legend swatch="var(--color-accent-500)" label="model call" />
        <Legend swatch="var(--color-good-500)" label="search / ranking" />
        <Legend swatch="var(--color-warn-500)" label="dashed = retry loop (bounded)" dashed />
      </div>
    </div>
  );
}

function LoopArrow({ fromIndex, toIndex, label }: { fromIndex: number; toIndex: number; label: string }) {
  const yFrom = TOP + fromIndex * (BOX_H + GAP) + BOX_H / 2;
  const yTo = TOP + toIndex * (BOX_H + GAP) + BOX_H / 2;
  const bulge = 34;
  return (
    <g>
      <path
        d={`M ${LEFT + BOX_W} ${yFrom} C ${LOOP_X + bulge} ${yFrom}, ${LOOP_X + bulge} ${yTo}, ${LEFT + BOX_W} ${yTo}`}
        fill="none" stroke="var(--color-warn-500)" strokeWidth={1.2}
        strokeDasharray="4 3" markerEnd="url(#arrow-loop)"
      />
      <text
        x={LOOP_X + bulge + 6} y={(yFrom + yTo) / 2 + 3}
        fontSize={10} fill="var(--color-warn-700)" fontFamily="var(--font-mono)"
      >
        {label}
      </text>
    </g>
  );
}

function Legend({ swatch, label, dashed }: { swatch: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="inline-block h-2 w-4 rounded-sm"
        style={dashed
          ? { border: `1.2px dashed ${swatch}` }
          : { background: swatch, opacity: 0.55, border: `1px solid ${swatch}` }}
      />
      {label}
    </span>
  );
}
