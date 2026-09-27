"use client";

import { useState } from "react";
import type { Citation } from "@/lib/types";
import { Badge, Card } from "./ui";

/**
 * The citation panel is the part that makes this a research tool rather than a
 * chatbot: every claim in the answer is traceable to the exact clause it came
 * from, and the user can read that clause without leaving the page.
 *
 * Two honesty details matter here. Excerpts the model did not actually cite are
 * shown separately and dimmed - they were retrieved but unused, which is useful
 * signal. And an estimated page number is rendered as "~p.7", never "p.7",
 * because the page was inferred from text length rather than read from a PDF.
 */
export function CitationList({ citations }: { citations: Citation[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [showUnused, setShowUnused] = useState(false);

  if (!citations.length) return null;
  const used = citations.filter((c) => c.used);
  const unused = citations.filter((c) => !c.used);
  const visible = showUnused ? citations : used.length ? used : citations;

  return (
    <div className="mt-3">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
          Sources
        </span>
        <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
          {used.length} cited · {citations.length} retrieved
        </span>
        {unused.length > 0 && (
          <button
            onClick={() => setShowUnused((v) => !v)}
            className="ml-auto text-[11.5px] underline underline-offset-2"
            style={{ color: "var(--text-muted)" }}
          >
            {showUnused ? "Hide unused" : `Show ${unused.length} unused`}
          </button>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        {visible.map((citation) => {
          const open = expanded === citation.index;
          return (
            <Card
              key={`${citation.chunk_id}-${citation.index}`}
              className="overflow-hidden"
              style={{ opacity: citation.used ? 1 : 0.62 }}
            >
              <button
                onClick={() => setExpanded(open ? null : citation.index)}
                className="flex w-full items-start gap-2.5 p-2.5 text-left"
                aria-expanded={open}
              >
                <span
                  className="mt-px grid h-5 w-5 shrink-0 place-items-center rounded-md text-[11px] font-semibold"
                  style={{
                    background: citation.used ? "var(--color-accent-100)" : "var(--surface-sunken)",
                    color: citation.used ? "var(--color-accent-700)" : "var(--text-muted)",
                  }}
                >
                  {citation.index}
                </span>

                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[12.5px] font-medium">{citation.title}</span>
                  <span className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[11px]" style={{ color: "var(--text-muted)" }}>
                    {citation.section_path && <span className="font-mono">§ {citation.section_path}</span>}
                    {citation.page_number != null && (
                      <span
                        title={
                          citation.page_confidence === "exact"
                            ? "Exact page from the PDF"
                            : "Estimated page — this document has no reliable page markers"
                        }
                      >
                        {citation.page_confidence === "exact" ? "p." : "~p."}
                        {citation.page_number}
                      </span>
                    )}
                    {citation.retriever && <Badge>{citation.retriever}</Badge>}
                    {typeof citation.score === "number" && (
                      <span className="font-mono tabular-nums">{citation.score.toFixed(3)}</span>
                    )}
                  </span>
                </span>

                <span className="mt-0.5 shrink-0 text-[11px]" style={{ color: "var(--text-muted)" }}>
                  {open ? "−" : "+"}
                </span>
              </button>

              {open && (
                <div className="border-t px-2.5 py-2.5" style={{ borderColor: "var(--border)" }}>
                  {citation.clause_types.length > 0 && (
                    <div className="mb-2 flex flex-wrap gap-1">
                      {citation.clause_types.map((slug) => (
                        <Badge key={slug} tone="accent">
                          {slug.replace(/_/g, " ")}
                        </Badge>
                      ))}
                    </div>
                  )}
                  <p
                    className="max-h-72 overflow-y-auto whitespace-pre-wrap text-[12.5px] leading-relaxed"
                    style={{ color: "var(--text)", fontFamily: "var(--font-serif)" }}
                  >
                    {citation.text}
                  </p>
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
