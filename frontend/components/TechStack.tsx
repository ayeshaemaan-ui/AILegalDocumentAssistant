"use client";

import { Card } from "./ui";

/** Every library that does real work, and what it is doing. */

type Entry = { name: string; role: string; note?: string };
type Group = { group: string; entries: Entry[] };

const STACK: Group[] = [
  {
    group: "Models",
    entries: [
      { name: "gpt-5", role: "Answer generation", note: "streamed; the only expensive call in the pipeline" },
      { name: "gpt-5-mini", role: "Routing, rewriting, grading, judging", note: "four roles, pennies each" },
      { name: "BAAI/bge-base-en-v1.5", role: "Dense embeddings", note: "768-d, runs locally on the GPU, free" },
      { name: "ms-marco-MiniLM-L-6-v2", role: "Cross-encoder re-ranker", note: "local, ~22M params" },
      { name: "claude-opus-5 / haiku-4-5", role: "Alternate provider", note: "same code path; one env var" },
    ],
  },
  {
    group: "Retrieval",
    entries: [
      { name: "chromadb", role: "Vector store (default)", note: "embedded, no server" },
      { name: "rank_bm25", role: "Sparse lexical arm", note: "Okapi BM25" },
      { name: "sentence-transformers", role: "Runs both local models" },
      { name: "torch (MPS)", role: "GPU inference on Apple Silicon", note: "serialised behind one process-wide lock" },
      { name: "langchain-postgres / pinecone / weaviate", role: "Alternate vector backends", note: "lazily imported" },
    ],
  },
  {
    group: "Orchestration",
    entries: [
      { name: "langgraph", role: "The workflow itself", note: "typed state, 11 nodes, 2 bounded retry loops" },
      { name: "langgraph-checkpoint-sqlite", role: "State persistence per thread" },
      { name: "langchain-core", role: "Retriever, tool and memory interfaces" },
      { name: "pydantic", role: "Every structured-output schema", note: "what makes graph edges safe to branch on" },
    ],
  },
  {
    group: "Ingestion",
    entries: [
      { name: "pdfplumber", role: "PDF text + exact page boundaries" },
      { name: "pypdf", role: "PDF fallback and metadata" },
      { name: "python-docx", role: "DOCX extraction" },
      { name: "langchain-text-splitters", role: "Recursive baseline splitter", note: "the structural splitter is hand-written" },
    ],
  },
  {
    group: "Serving",
    entries: [
      { name: "fastapi", role: "26 endpoints" },
      { name: "sse-starlette", role: "Token streaming transport" },
      { name: "uvicorn", role: "ASGI server" },
      { name: "next.js 16 / react 19", role: "This interface" },
      { name: "tailwind 4", role: "Design tokens, light + dark" },
    ],
  },
  {
    group: "Evaluation & ops",
    entries: [
      { name: "CUAD v1", role: "510 contracts, lawyer-annotated", note: "the ground truth everything is scored against" },
      { name: "pytest", role: "184 tests" },
      { name: "opentelemetry", role: "Spans across LLM and non-LLM work" },
      { name: "langsmith", role: "Prompt-level LLM tracing", note: "optional" },
      { name: "helicone", role: "Proxy-level logging", note: "optional, zero code at call sites" },
      { name: "sqlite", role: "Sessions, cost ledger, traces, eval runs" },
      { name: "docker / compose", role: "Multi-stage images, non-root" },
    ],
  },
];

export function TechStack() {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {STACK.map((g) => (
        <Card key={g.group} className="p-4">
          <h3 className="mb-2.5 text-[13.5px] font-semibold">{g.group}</h3>
          <div className="flex flex-col gap-2">
            {g.entries.map((e) => (
              <div key={e.name} className="text-[12.5px] leading-snug">
                <span className="font-mono text-[11.5px] font-medium">{e.name}</span>
                <span className="mx-1.5" style={{ color: "var(--text-muted)" }}>—</span>
                <span>{e.role}</span>
                {e.note && (
                  <span className="block text-[11.5px] italic" style={{ color: "var(--text-muted)" }}>
                    {e.note}
                  </span>
                )}
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  );
}
