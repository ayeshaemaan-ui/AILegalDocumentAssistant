"use client";

import { Card } from "./ui";

/**
 * Stage-by-stage breakdown. Each row answers three separate questions that
 * usually get blurred together in architecture docs: what *technique* is being
 * applied, what *technology* runs it, and what *algorithm* sits underneath.
 */

type Stage = {
  n: number;
  stage: string;
  file: string;
  technique: string;
  technology: string;
  algorithm: string;
  why: string;
};

const STAGES: Stage[] = [
  {
    n: 1,
    stage: "Guard input",
    file: "core/guardrails.py",
    technique: "Input validation · prompt-injection screening",
    technology: "Compiled regex (no model)",
    algorithm: "Pattern matching against an injection + advice-seeking corpus",
    why: "Deterministic, free, and instant. Stops a bad request before it costs a model call, and cannot itself be talked out of its rules.",
  },
  {
    n: 2,
    stage: "Route intent",
    file: "graph/nodes.py · route",
    technique: "Zero-shot classification with a constrained schema",
    technology: "gpt-5-mini · structured output",
    algorithm: "Schema-constrained decoding into a 6-value enum",
    why: "Legal-advice requests must be declined, not answered. A constrained schema means the model physically cannot return a branch the router does not understand.",
  },
  {
    n: 3,
    stage: "Rewrite query",
    file: "graph/nodes.py · rewrite_query",
    technique: "One-shot query rewriting · HyDE-style term expansion",
    technology: "gpt-5-mini",
    algorithm: "Conversational phrasing → contract vocabulary + clause-type filter proposal",
    why: "Users type \"can I quit early\"; contracts say \"termination for convenience\". This is also what gives BM25 a keyword query instead of a sentence.",
  },
  {
    n: 4,
    stage: "Retrieve",
    file: "retrieval/retriever.py",
    technique: "Hybrid dense + sparse retrieval",
    technology: "BAAI/bge-base-en-v1.5 (768-d) · rank_bm25 · Chroma",
    algorithm: "Cosine similarity ‖ BM25 Okapi → Reciprocal Rank Fusion (k=60)",
    why: "Embeddings find meaning, BM25 finds identifiers. Contract questions need both — a party name and \"what are my exit rights\" fail on opposite retrievers.",
  },
  {
    n: 5,
    stage: "Re-rank",
    file: "retrieval/rerank.py",
    technique: "Cross-encoder precision pass",
    technology: "cross-encoder/ms-marco-MiniLM-L-6-v2",
    algorithm: "Joint BERT forward pass over (query, clause) pairs → sigmoid",
    why: "The bi-encoder never sees query and clause together, so it misses who an obligation binds. The cross-encoder does — too slowly for 4,200 chunks, so it only scores the top 20.",
  },
  {
    n: 6,
    stage: "Grade context",
    file: "graph/nodes.py · grade_context",
    technique: "Self-critique gate before generation",
    technology: "gpt-5-mini · structured output",
    algorithm: "Sufficiency classification → conditional edge back to retrieve",
    why: "A cheap call that decides whether to spend an expensive one. If the excerpts cannot answer, widen the search rather than generating from nothing.",
  },
  {
    n: 7,
    stage: "Generate",
    file: "graph/nodes.py · generate",
    technique: "Grounded generation · few-shot · chain of thought · role prompting",
    technology: "gpt-5 · token streaming",
    algorithm: "Context injection with a bracketed-citation output contract",
    why: "Four few-shot examples each teach a different failure mode — found, partial, absent, advice-seeking. The \"absent\" one is what stops the model inventing a clause.",
  },
  {
    n: 8,
    stage: "Verify",
    file: "graph/nodes.py · verify",
    technique: "Citation validation + groundedness audit",
    technology: "Regex, then gpt-5-mini",
    algorithm: "Citation-index set check → claim-level entailment → conditional regenerate",
    why: "A model citing [7] when given 6 excerpts has fabricated evidence. That is a parsing problem and is caught for free, before any LLM judge is involved.",
  },
  {
    n: 9,
    stage: "Deliver",
    file: "api/routes_chat.py",
    technique: "Server-Sent Events streaming",
    technology: "FastAPI · sse-starlette → React 19",
    algorithm: "asyncio.Queue drained into SSE frames",
    why: "One-directional, so SSE beats WebSockets: plain HTTP, survives proxies, reconnects itself. LangGraph runs nodes to completion, so a queue is what lets tokens escape mid-node.",
  },
];

export function StageTable() {
  return (
    <div className="flex flex-col gap-2.5">
      {STAGES.map((s) => (
        <Card key={s.n} className="p-3.5">
          <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <span
              className="grid h-5 w-5 shrink-0 place-items-center rounded-md text-[11px] font-semibold"
              style={{ background: "var(--color-accent-100)", color: "var(--color-accent-700)" }}
            >
              {s.n}
            </span>
            <h3 className="text-[13.5px] font-semibold">{s.stage}</h3>
            <span className="font-mono text-[10.5px]" style={{ color: "var(--text-muted)" }}>
              {s.file}
            </span>
          </div>

          <dl className="mt-2 grid gap-x-4 gap-y-1.5 sm:grid-cols-[92px_1fr]">
            <Row label="Technique" value={s.technique} />
            <Row label="Technology" value={s.technology} mono />
            <Row label="Algorithm" value={s.algorithm} mono />
          </dl>

          <p className="mt-2 text-[12.5px] italic leading-relaxed" style={{ color: "var(--text-muted)" }}>
            {s.why}
          </p>
        </Card>
      ))}
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <>
      <dt className="text-[10.5px] font-semibold uppercase tracking-wide sm:pt-0.5" style={{ color: "var(--text-muted)" }}>
        {label}
      </dt>
      <dd className={`text-[12.5px] leading-relaxed ${mono ? "font-mono text-[11.5px]" : ""}`}>{value}</dd>
    </>
  );
}
