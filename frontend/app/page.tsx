"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MessageBubble } from "@/components/MessageBubble";
import { FilterSidebar } from "@/components/FilterSidebar";
import { Button, EmptyState } from "@/components/ui";
import { TechStrip } from "@/components/TechStrip";
import { api, getStoredSession, storeSession, streamAsk } from "@/lib/api";
import type { AskResponse, ChatMessage, Filters } from "@/lib/types";

const SUGGESTIONS = [
  "What are the termination rights in these agreements?",
  "Which contracts are governed by Delaware law?",
  "Is there a cap on liability, and what is it?",
  "What exclusivity commitments do the distributor agreements impose?",
];

const EMPTY_FILTERS: Filters = { doc_ids: [], clause_types: [], contract_types: [], governing_law: null };

export default function ResearchPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [mode, setMode] = useState<"rag" | "agent">("rag");

  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => setSessionId(getStoredSession()), []);

  // Keep the newest message in view as tokens stream in.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const patchLast = useCallback((patch: Partial<ChatMessage> | ((m: ChatMessage) => Partial<ChatMessage>)) => {
    setMessages((prev) => {
      if (!prev.length) return prev;
      const next = [...prev];
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, ...(typeof patch === "function" ? patch(last) : patch) };
      return next;
    });
  }, []);

  async function send(question: string) {
    const trimmed = question.trim();
    if (!trimmed || busy) return;

    setInput("");
    setBusy(true);
    setMessages((prev) => [
      ...prev,
      { id: `u-${Date.now()}`, role: "user", content: trimmed },
      { id: `a-${Date.now()}`, role: "assistant", content: "", streaming: true },
    ]);

    // The agent path is a different backend shape (tool loop, no token stream),
    // so it is handled separately rather than forced through the SSE client.
    if (mode === "agent") {
      try {
        const result = await api.runAgent(trimmed, "agent", sessionId);
        if (result.session_id) {
          setSessionId(result.session_id);
          storeSession(result.session_id);
        }
        patchLast({ content: result.answer, streaming: false });
      } catch (error) {
        patchLast({
          streaming: false,
          error: error instanceof Error ? error.message : "The agent request failed.",
        });
      } finally {
        setBusy(false);
      }
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamAsk(trimmed, filters, sessionId, {
        onMeta: ({ session_id }) => {
          setSessionId(session_id);
          storeSession(session_id);
        },
        onToken: (text) => patchLast((m) => ({ content: m.content + text })),
        onThinking: (text) => patchLast((m) => ({ thinking: (m.thinking ?? "") + text })),
        // The graph regenerates when an answer fails its groundedness check.
        // Clearing the buffer prevents the retry being appended to the draft.
        onRestart: () => patchLast({ content: "" }),
        onDone: (data: AskResponse) =>
          patchLast((m) => ({
            streaming: false,
            meta: data,
            // A refusal or meta answer never streams tokens - take the text
            // from the final envelope instead of leaving the bubble empty.
            content: data.streamed && m.content ? m.content : data.answer,
          })),
        onError: (data) => patchLast({ streaming: false, error: data.message }),
      }, controller.signal);
    } catch (error) {
      if ((error as Error)?.name !== "AbortError") {
        patchLast({
          streaming: false,
          error: error instanceof Error ? error.message : "The request failed.",
        });
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  function stop() {
    abortRef.current?.abort();
    patchLast({ streaming: false });
    setBusy(false);
  }

  function newConversation() {
    abortRef.current?.abort();
    setMessages([]);
    setSessionId(null);
    try {
      localStorage.removeItem("legallens.session");
    } catch {
      /* non-fatal */
    }
  }

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      <div className="hidden lg:flex">
        <FilterSidebar filters={filters} onChange={setFilters} />
      </div>

      <div className="flex min-w-0 flex-1 flex-col">
        <div
          className="flex items-center gap-2 border-b px-4 py-2"
          style={{ borderColor: "var(--border)", background: "var(--surface)" }}
        >
          <div className="flex rounded-lg border p-0.5" style={{ borderColor: "var(--border)" }}>
            {(["rag", "agent"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                title={
                  m === "rag"
                    ? "Single-pass RAG: route, retrieve, generate, verify. Streams tokens."
                    : "Multi-step agent: chooses its own searches via tool calling. Best for cross-document questions."
                }
                className="rounded-[6px] px-2.5 py-1 text-[12px] font-medium transition-colors"
                style={
                  mode === m
                    ? { background: "var(--color-accent-600)", color: "#fff" }
                    : { color: "var(--text-muted)" }
                }
              >
                {m === "rag" ? "RAG" : "Agent"}
              </button>
            ))}
          </div>
          <TechStrip
            className="hidden xl:flex"
            items={["hybrid retrieval", "RRF", "cross-encoder", "LangGraph", "SSE streaming"]}
          />
          <span className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>
            {filters.doc_ids.length > 0
              ? `Scoped to ${filters.doc_ids.length} document${filters.doc_ids.length > 1 ? "s" : ""}`
              : "Searching all indexed documents"}
          </span>
          <div className="ml-auto flex items-center gap-2">
            {busy && <Button variant="ghost" onClick={stop}>Stop</Button>}
            {messages.length > 0 && (
              <Button variant="secondary" onClick={newConversation}>
                New
              </Button>
            )}
          </div>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="mx-auto flex max-w-3xl flex-col gap-6 px-5 py-6">
            {messages.length === 0 ? (
              <EmptyState
                title="Ask a question about your contracts"
                body="LegalLens searches the indexed documents, retrieves the clauses that answer your question, and cites each one. It reports what the documents say — it does not give legal advice."
                action={
                  <div className="flex flex-col gap-1.5">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s}
                        onClick={() => send(s)}
                        className="rounded-lg border px-3 py-2 text-left text-[13px] transition-opacity hover:opacity-75"
                        style={{ borderColor: "var(--border)", background: "var(--surface)" }}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                }
              />
            ) : (
              messages.map((message) => <MessageBubble key={message.id} message={message} />)
            )}
          </div>
        </div>

        <div className="border-t px-5 py-3" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <div className="mx-auto max-w-3xl">
            <div
              className="flex items-end gap-2 rounded-xl border px-3 py-2"
              style={{ borderColor: "var(--border)", background: "var(--bg)" }}
            >
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  const el = e.target as HTMLTextAreaElement;
                  el.style.height = "auto";
                  el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
                }}
                onKeyDown={(e) => {
                  // Enter sends; Shift+Enter is a newline, as in every chat app.
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send(input);
                  }
                }}
                placeholder="Ask about a clause, obligation, date, or restriction…"
                className="max-h-48 flex-1 resize-none bg-transparent text-[14px] leading-relaxed outline-none"
                style={{ color: "var(--text)" }}
                disabled={busy}
              />
              <Button variant="primary" onClick={() => send(input)} disabled={busy || !input.trim()}>
                {busy ? "…" : "Ask"}
              </Button>
            </div>
            <p className="mt-1.5 text-center text-[11px]" style={{ color: "var(--text-muted)" }}>
              Answers are drawn only from indexed documents and cite their source. Not a substitute for a qualified lawyer.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
