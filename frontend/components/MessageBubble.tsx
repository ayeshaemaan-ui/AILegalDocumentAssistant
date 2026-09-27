"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "@/lib/types";
import { CitationList } from "./CitationList";
import { PipelineTrace } from "./PipelineTrace";
import { Badge, Disclosure } from "./ui";

function ConfidenceBadge({ message }: { message: ChatMessage }) {
  const verdict = message.meta?.groundedness?.verdict;
  const ratio = message.meta?.groundedness?.grounded_ratio;

  if (verdict === "grounded")
    return (
      <Badge tone="good" title="Every claim in this answer was matched to a retrieved excerpt.">
        grounded
      </Badge>
    );
  if (verdict === "partially_grounded")
    return (
      <Badge
        tone="warn"
        title={`${Math.round((ratio ?? 0) * 100)}% of claims matched a retrieved excerpt. Verify the rest against the cited clauses.`}
      >
        partly grounded
      </Badge>
    );
  if (verdict === "hallucinated")
    return (
      <Badge tone="bad" title="Claims in this answer were not supported by the retrieved excerpts.">
        unverified
      </Badge>
    );
  return null;
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div
          className="max-w-[80%] rounded-2xl rounded-br-md px-3.5 py-2.5 text-[14px] leading-relaxed"
          style={{ background: "var(--color-accent-600)", color: "#fff" }}
        >
          {message.content}
        </div>
      </div>
    );
  }

  const meta = message.meta;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
          LegalLens
        </span>
        <ConfidenceBadge message={message} />
        {meta?.blocked && <Badge tone="warn">declined</Badge>}
      </div>

      {message.thinking && (
        <Disclosure label="Reasoning">
          <div
            className="max-h-52 overflow-y-auto whitespace-pre-wrap rounded-lg border p-2.5 text-[12px] leading-relaxed"
            style={{ borderColor: "var(--border)", background: "var(--surface-sunken)", color: "var(--text-muted)" }}
          >
            {message.thinking}
          </div>
        </Disclosure>
      )}

      <div className={`prose-answer text-[14px] ${message.streaming ? "caret" : ""}`}>
        {message.error ? (
          <p style={{ color: "var(--color-bad-500)" }}>{message.error}</p>
        ) : message.content ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        ) : message.streaming ? (
          <span style={{ color: "var(--text-muted)" }}>Searching the contracts…</span>
        ) : null}
      </div>

      {meta && !message.streaming && (
        <>
          {meta.citations?.length > 0 && <CitationList citations={meta.citations} />}
          <div className="mt-1.5">
            <PipelineTrace meta={meta} />
          </div>
        </>
      )}
    </div>
  );
}
