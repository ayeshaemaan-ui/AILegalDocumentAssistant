"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { ClauseType, DocumentSummary, Filters } from "@/lib/types";
import { Badge, Spinner } from "./ui";

/**
 * Metadata filters (Tier 1 M1). Narrowing to one document or one clause type
 * turns a fuzzy search over 4,000 chunks into a precise one over a handful,
 * which is usually the difference between a vague answer and an exact clause.
 */
export function FilterSidebar({
  filters,
  onChange,
}: {
  filters: Filters;
  onChange: (next: Filters) => void;
}) {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [clauseTypes, setClauseTypes] = useState<ClauseType[]>([]);
  const [loading, setLoading] = useState(true);
  const [docQuery, setDocQuery] = useState("");
  const [clauseQuery, setClauseQuery] = useState("");

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.documents(), api.clauseTypes()])
      .then(([docs, clauses]) => {
        if (cancelled) return;
        setDocuments(docs);
        setClauseTypes(clauses.clause_types);
      })
      .catch(() => void 0)
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredDocs = useMemo(() => {
    const q = docQuery.trim().toLowerCase();
    const list = q ? documents.filter((d) => d.title.toLowerCase().includes(q)) : documents;
    return list.slice(0, 60);
  }, [documents, docQuery]);

  const filteredClauses = useMemo(() => {
    const q = clauseQuery.trim().toLowerCase();
    return q ? clauseTypes.filter((c) => c.label.toLowerCase().includes(q)) : clauseTypes;
  }, [clauseTypes, clauseQuery]);

  function toggle(key: "doc_ids" | "clause_types", value: string) {
    const current = filters[key];
    onChange({
      ...filters,
      [key]: current.includes(value) ? current.filter((v) => v !== value) : [...current, value],
    });
  }

  const activeCount = filters.doc_ids.length + filters.clause_types.length;

  return (
    <aside
      className="flex w-72 shrink-0 flex-col border-r"
      style={{ borderColor: "var(--border)", background: "var(--surface)" }}
    >
      <div className="flex items-center gap-2 border-b px-3.5 py-2.5" style={{ borderColor: "var(--border)" }}>
        <span className="text-[12.5px] font-semibold">Filters</span>
        {activeCount > 0 && (
          <>
            <Badge tone="accent">{activeCount}</Badge>
            <button
              onClick={() => onChange({ doc_ids: [], clause_types: [], contract_types: [], governing_law: null })}
              className="ml-auto text-[11.5px] underline underline-offset-2"
              style={{ color: "var(--text-muted)" }}
            >
              Clear
            </button>
          </>
        )}
      </div>

      {loading ? (
        <div className="flex items-center gap-2 p-3.5 text-[12.5px]" style={{ color: "var(--text-muted)" }}>
          <Spinner /> Loading corpus…
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          <section className="border-b px-3.5 py-3" style={{ borderColor: "var(--border)" }}>
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                Documents
              </span>
              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                {documents.length}
              </span>
            </div>
            <input
              value={docQuery}
              onChange={(e) => setDocQuery(e.target.value)}
              placeholder="Search documents…"
              className="mb-1.5 w-full rounded-md border px-2 py-1.5 text-[12.5px] outline-none"
              style={{ borderColor: "var(--border)", background: "var(--bg)", color: "var(--text)" }}
            />
            <div className="flex max-h-64 flex-col gap-0.5 overflow-y-auto">
              {filteredDocs.map((doc) => (
                <label
                  key={doc.doc_id}
                  className="flex cursor-pointer items-start gap-2 rounded-md px-1.5 py-1 text-[12px] hover:opacity-80"
                >
                  <input
                    type="checkbox"
                    className="mt-0.5 shrink-0"
                    checked={filters.doc_ids.includes(doc.doc_id)}
                    onChange={() => toggle("doc_ids", doc.doc_id)}
                  />
                  <span className="min-w-0">
                    <span className="block truncate" title={doc.title}>
                      {doc.title}
                    </span>
                    <span className="text-[10.5px]" style={{ color: "var(--text-muted)" }}>
                      {doc.contract_type ?? "Agreement"}
                      {doc.governing_law ? ` · ${doc.governing_law}` : ""} · {doc.chunk_count} chunks
                    </span>
                  </span>
                </label>
              ))}
              {!filteredDocs.length && (
                <span className="px-1.5 py-2 text-[12px]" style={{ color: "var(--text-muted)" }}>
                  No documents match.
                </span>
              )}
            </div>
          </section>

          <section className="px-3.5 py-3">
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                Clause type
              </span>
              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                {clauseTypes.length}
              </span>
            </div>
            <input
              value={clauseQuery}
              onChange={(e) => setClauseQuery(e.target.value)}
              placeholder="Search clause types…"
              className="mb-1.5 w-full rounded-md border px-2 py-1.5 text-[12.5px] outline-none"
              style={{ borderColor: "var(--border)", background: "var(--bg)", color: "var(--text)" }}
            />
            <div className="flex max-h-72 flex-col gap-0.5 overflow-y-auto">
              {filteredClauses.map((clause) => (
                <label
                  key={clause.slug}
                  title={clause.description}
                  className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 text-[12px] hover:opacity-80"
                >
                  <input
                    type="checkbox"
                    className="shrink-0"
                    checked={filters.clause_types.includes(clause.slug)}
                    onChange={() => toggle("clause_types", clause.slug)}
                  />
                  <span className="truncate">{clause.label}</span>
                </label>
              ))}
            </div>
          </section>
        </div>
      )}
    </aside>
  );
}
