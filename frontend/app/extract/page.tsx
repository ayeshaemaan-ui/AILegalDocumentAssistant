"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { ClauseType, DocumentSummary } from "@/lib/types";
import { Badge, Button, Card, EmptyState, Spinner } from "@/components/ui";
import { TechStrip } from "@/components/TechStrip";

/**
 * Clause matrix: contracts down the side, clause types across the top.
 *
 * The chat interface answers one question at a time. This is for the other half
 * of contract review — the same twelve answers out of forty agreements, in a
 * grid, exported to a spreadsheet. It is the shape of CUAD's own
 * master_clauses.csv, which is not a coincidence: it is what contract review
 * actually produces.
 */

// Sensible opening selection — the clauses a reviewer almost always wants.
const DEFAULT_CLAUSES = ["governing_law", "termination_for_convenience", "cap_on_liability"];

const STATUS_STYLE: Record<string, { bg: string; fg: string; title: string }> = {
  found: { bg: "var(--color-good-100)", fg: "var(--color-good-700)", title: "Stated in the contract" },
  not_present: { bg: "var(--surface-sunken)", fg: "var(--text-muted)", title: "The contract has no such clause" },
  unsure: { bg: "var(--color-warn-100)", fg: "var(--color-warn-700)", title: "Ambiguous — needs review" },
  error: { bg: "var(--color-bad-100)", fg: "var(--color-bad-700)", title: "Extraction failed" },
};

export default function ExtractPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [clauseTypes, setClauseTypes] = useState<ClauseType[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [selectedClauses, setSelectedClauses] = useState<string[]>(DEFAULT_CLAUSES);
  const [job, setJob] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [docQuery, setDocQuery] = useState("");
  const [clauseQuery, setClauseQuery] = useState("");
  const [detail, setDetail] = useState<any>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    Promise.all([api.documents(), api.clauseTypes()])
      .then(([docs, clauses]) => {
        setDocuments(docs);
        setClauseTypes(clauses.clause_types);
        setSelectedDocs(docs.slice(0, 5).map((d) => d.doc_id));
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load the corpus."))
      .finally(() => setLoading(false));
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, []);

  const poll = useCallback(async (jobId: string) => {
    try {
      const status = await api.matrixJob(jobId);
      setJob(status);
      if (status.status === "running" || status.status === "queued") {
        pollTimer.current = setTimeout(() => poll(jobId), 1200);
      }
    } catch {
      /* leave the last known state on screen */
    }
  }, []);

  async function run() {
    setError(null);
    setDetail(null);
    try {
      const started = await api.startMatrix({
        doc_ids: selectedDocs,
        clause_slugs: selectedClauses,
      });
      setJob(started);
      poll(started.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start extraction.");
    }
  }

  const cellIndex = useMemo(() => {
    const map = new Map<string, any>();
    for (const c of job?.cells ?? []) map.set(`${c.doc_id}|${c.clause_slug}`, c);
    return map;
  }, [job]);

  const filteredDocs = useMemo(() => {
    const q = docQuery.trim().toLowerCase();
    return q ? documents.filter((d) => d.title.toLowerCase().includes(q)) : documents;
  }, [documents, docQuery]);

  const filteredClauses = useMemo(() => {
    const q = clauseQuery.trim().toLowerCase();
    return q ? clauseTypes.filter((c) => c.label.toLowerCase().includes(q)) : clauseTypes;
  }, [clauseTypes, clauseQuery]);

  const cellCount = selectedDocs.length * selectedClauses.length;
  const running = job && (job.status === "running" || job.status === "queued");
  const done = job?.status === "completed";

  function toggle(list: string[], setList: (v: string[]) => void, value: string) {
    setList(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);
  }

  return (
    <div className="mx-auto max-w-[1400px] px-5 py-7">
      <header className="mb-5">
        <h1 className="text-[20px] font-semibold tracking-tight">Extract</h1>
        <p className="mt-1 max-w-3xl text-[13.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
          Pull the same clauses out of many contracts at once. Every cell is retrieved, extracted and
          cited independently — then exported as CSV.
        </p>
        <TechStrip
          className="mt-2"
          items={["metadata filtering", "hybrid retrieval", "structured outputs", "CSV export"]}
        />
      </header>

      {loading ? (
        <Card className="flex items-center gap-2 p-5 text-[13px]" style={{ color: "var(--text-muted)" }}>
          <Spinner /> Loading corpus…
        </Card>
      ) : (
        <>
          <div className="mb-4 grid gap-3 lg:grid-cols-2">
            <Selector
              title="Contracts"
              count={selectedDocs.length}
              total={documents.length}
              query={docQuery}
              onQuery={setDocQuery}
              onAll={() => setSelectedDocs(filteredDocs.slice(0, 40).map((d) => d.doc_id))}
              onNone={() => setSelectedDocs([])}
            >
              {filteredDocs.slice(0, 80).map((doc) => (
                <label key={doc.doc_id} className="flex cursor-pointer items-start gap-2 rounded-md px-1.5 py-1 text-[12px]">
                  <input
                    type="checkbox"
                    className="mt-0.5 shrink-0"
                    checked={selectedDocs.includes(doc.doc_id)}
                    onChange={() => toggle(selectedDocs, setSelectedDocs, doc.doc_id)}
                  />
                  <span className="min-w-0">
                    <span className="block truncate" title={doc.title}>{doc.title}</span>
                    <span className="text-[10.5px]" style={{ color: "var(--text-muted)" }}>
                      {doc.contract_type ?? "Agreement"}
                    </span>
                  </span>
                </label>
              ))}
            </Selector>

            <Selector
              title="Clause types"
              count={selectedClauses.length}
              total={clauseTypes.length}
              query={clauseQuery}
              onQuery={setClauseQuery}
              onAll={() => setSelectedClauses(filteredClauses.slice(0, 12).map((c) => c.slug))}
              onNone={() => setSelectedClauses([])}
            >
              {filteredClauses.map((clause) => (
                <label
                  key={clause.slug}
                  title={clause.description}
                  className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 text-[12px]"
                >
                  <input
                    type="checkbox"
                    className="shrink-0"
                    checked={selectedClauses.includes(clause.slug)}
                    onChange={() => toggle(selectedClauses, setSelectedClauses, clause.slug)}
                  />
                  <span className="truncate">{clause.label}</span>
                </label>
              ))}
            </Selector>
          </div>

          <Card className="mb-4 p-4">
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="primary" onClick={run} disabled={running || !cellCount}>
                {running ? "Extracting…" : "Run extraction"}
              </Button>
              <span className="text-[12.5px]" style={{ color: "var(--text-muted)" }}>
                {selectedDocs.length} contracts × {selectedClauses.length} clauses ={" "}
                <strong style={{ color: cellCount > 600 ? "var(--color-bad-500)" : "var(--text)" }}>
                  {cellCount} cells
                </strong>
                {cellCount > 600 && " — over the 600 limit"}
              </span>
              {done && (
                <a
                  href={api.matrixCsvUrl(job.job_id)}
                  download
                  className="ml-auto inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[13px] font-medium"
                  style={{ borderColor: "var(--border)", color: "var(--text)" }}
                >
                  ↓ Download CSV
                </a>
              )}
            </div>

            {job && (
              <div className="mt-3">
                <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
                  {running && <Spinner />}
                  <span className="font-medium">
                    {done ? "Complete" : job.status === "failed" ? "Failed" : `${job.processed}/${job.total} cells`}
                  </span>
                  {job.summary && (
                    <span className="flex flex-wrap gap-1">
                      <Badge tone="good">{job.summary.found} found</Badge>
                      <Badge>{job.summary.not_present} absent</Badge>
                      <Badge tone="warn">{job.summary.unsure} unsure</Badge>
                      {job.summary.error > 0 && <Badge tone="bad">{job.summary.error} error</Badge>}
                    </span>
                  )}
                  <span style={{ color: "var(--text-muted)" }}>
                    {job.elapsed_s}s · ${Number(job.cost_usd ?? 0).toFixed(4)}
                  </span>
                </div>
                {running && job.total > 0 && (
                  <div className="mt-2 h-1 overflow-hidden rounded-full" style={{ background: "var(--border)" }}>
                    <div
                      className="h-full rounded-full transition-all"
                      style={{ width: `${(job.processed / job.total) * 100}%`, background: "var(--color-accent-500)" }}
                    />
                  </div>
                )}
                {job.error && (
                  <p className="mt-2 text-[12.5px]" style={{ color: "var(--color-bad-500)" }}>{job.error}</p>
                )}
              </div>
            )}

            {error && (
              <p className="mt-2.5 text-[12.5px]" style={{ color: "var(--color-bad-500)" }}>{error}</p>
            )}
          </Card>

          {job?.cells?.length ? (
            <Card className="overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-[12.5px]">
                  <thead>
                    <tr className="border-b text-left" style={{ borderColor: "var(--border)" }}>
                      <th className="sticky left-0 z-10 px-3 py-2 font-medium" style={{ background: "var(--surface)", color: "var(--text-muted)" }}>
                        Contract
                      </th>
                      {job.clause_slugs.map((slug: string) => (
                        <th key={slug} className="px-3 py-2 font-medium" style={{ color: "var(--text-muted)" }}>
                          {job.clause_labels?.[slug] ?? slug}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {job.doc_ids.map((docId: string) => (
                      <tr key={docId} className="border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                        <td
                          className="sticky left-0 z-10 max-w-[260px] px-3 py-2"
                          style={{ background: "var(--surface)" }}
                        >
                          <span className="block truncate" title={job.titles?.[docId]}>
                            {job.titles?.[docId] ?? docId}
                          </span>
                        </td>
                        {job.clause_slugs.map((slug: string) => {
                          const cell = cellIndex.get(`${docId}|${slug}`);
                          const status = cell?.error ? "error" : cell?.status ?? "pending";
                          const style = STATUS_STYLE[status];
                          return (
                            <td key={slug} className="px-3 py-2">
                              {cell ? (
                                <button
                                  onClick={() => setDetail({ ...cell, title: job.titles?.[docId], label: job.clause_labels?.[slug] })}
                                  title={style?.title}
                                  className="max-w-[210px] truncate rounded-md px-1.5 py-0.5 text-left"
                                  style={{ background: style?.bg, color: style?.fg }}
                                >
                                  {cell.error ? "error" : cell.status === "found" ? (cell.value || "found") : cell.status === "not_present" ? "—" : "?"}
                                </button>
                              ) : (
                                <span className="shimmer inline-block h-4 w-14 rounded" style={{ background: "var(--surface-sunken)" }} />
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ) : job ? null : (
            <Card>
              <EmptyState
                title="Choose contracts and clause types"
                body="Each cell runs its own filtered retrieval and extraction, so every value comes back with the section it was taken from."
              />
            </Card>
          )}

          {detail && (
            <Card className="mt-4 p-4">
              <div className="mb-2 flex items-start gap-2">
                <div>
                  <h3 className="text-[14px] font-semibold">{detail.label}</h3>
                  <p className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>{detail.title}</p>
                </div>
                <button onClick={() => setDetail(null)} className="ml-auto text-[12px] underline underline-offset-2" style={{ color: "var(--text-muted)" }}>
                  Close
                </button>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Badge tone={detail.status === "found" ? "good" : detail.status === "unsure" ? "warn" : "neutral"}>
                  {detail.status}
                </Badge>
                <Badge>confidence: {detail.confidence}</Badge>
                {detail.section_path && <Badge>§ {detail.section_path}</Badge>}
                {detail.page_number != null && (
                  <Badge>{detail.page_confidence === "exact" ? "p." : "~p."}{detail.page_number}</Badge>
                )}
              </div>
              {detail.value && (
                <p className="mt-2.5 text-[13px]"><strong>Value:</strong> {detail.value}</p>
              )}
              {detail.quote && (
                <blockquote
                  className="mt-2 rounded-lg border-l-2 px-3 py-2 text-[12.5px] leading-relaxed"
                  style={{ borderColor: "var(--color-accent-400)", background: "var(--surface-sunken)", fontFamily: "var(--font-serif)" }}
                >
                  {detail.quote}
                </blockquote>
              )}
              {detail.error && (
                <p className="mt-2 text-[12.5px]" style={{ color: "var(--color-bad-500)" }}>{detail.error}</p>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function Selector({
  title, count, total, query, onQuery, onAll, onNone, children,
}: {
  title: string; count: number; total: number; query: string;
  onQuery: (v: string) => void; onAll: () => void; onNone: () => void; children: React.ReactNode;
}) {
  return (
    <Card className="p-3.5">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-[12.5px] font-semibold">{title}</span>
        <Badge tone={count ? "accent" : "neutral"}>{count} of {total}</Badge>
        <span className="ml-auto flex gap-2 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
          <button onClick={onAll} className="underline underline-offset-2">Select shown</button>
          <button onClick={onNone} className="underline underline-offset-2">Clear</button>
        </span>
      </div>
      <input
        value={query}
        onChange={(e) => onQuery(e.target.value)}
        placeholder={`Search ${title.toLowerCase()}…`}
        className="mb-1.5 w-full rounded-md border px-2 py-1.5 text-[12.5px] outline-none"
        style={{ borderColor: "var(--border)", background: "var(--bg)", color: "var(--text)" }}
      />
      <div className="flex max-h-56 flex-col gap-0.5 overflow-y-auto">{children}</div>
    </Card>
  );
}
