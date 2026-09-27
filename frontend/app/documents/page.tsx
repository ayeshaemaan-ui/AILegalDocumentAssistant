"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { DocumentSummary } from "@/lib/types";
import { Badge, Button, Card, EmptyState, Spinner, Stat } from "@/components/ui";
import { TechStrip } from "@/components/TechStrip";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [job, setJob] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [docs, s] = await Promise.all([api.documents(), api.documentStats()]);
      setDocuments(docs);
      setStats(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load documents.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, [refresh]);

  /** Indexing runs server-side in the background, so progress is polled. */
  const pollJob = useCallback(
    async (jobId: string) => {
      try {
        const status = await api.ingestJob(jobId);
        setJob(status);
        if (status.status === "running" || status.status === "queued") {
          pollTimer.current = setTimeout(() => pollJob(jobId), 1200);
        } else {
          await refresh();
        }
      } catch {
        setJob(null);
      }
    },
    [refresh],
  );

  async function handleUpload(file: File) {
    setError(null);
    try {
      const result = await api.upload(file);
      setJob({ ...result, status: "queued" });
      pollJob(result.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    }
  }

  async function handleBulkIngest(limit: number) {
    setError(null);
    try {
      const result = await api.ingest(limit);
      setJob({ ...result, status: "queued", processed: 0 });
      pollJob(result.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ingestion failed to start.");
    }
  }

  async function handleDelete(docId: string, title: string) {
    if (!confirm(`Remove "${title}" from the index? This deletes its vectors and cannot be undone.`)) return;
    try {
      await api.deleteDocument(docId);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed.");
    }
  }

  const filtered = query.trim()
    ? documents.filter((d) => d.title.toLowerCase().includes(query.trim().toLowerCase()))
    : documents;

  const running = job && (job.status === "running" || job.status === "queued");

  return (
    <div className="mx-auto max-w-[1200px] px-5 py-7">
      <header className="mb-5">
        <h1 className="text-[20px] font-semibold tracking-tight">Documents</h1>
        <p className="mt-1 text-[13.5px]" style={{ color: "var(--text-muted)" }}>
          Every document here has been parsed, split into clause-level chunks, tagged against the
          41-category clause taxonomy, and embedded into the vector index.
        </p>
        <TechStrip
          className="mt-2"
          items={["pdfplumber", "structural chunking", "BGE-base-en-v1.5", "Chroma", "BM25"]}
        />
      </header>

      {stats && (
        <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Documents" value={stats.documents ?? 0} />
          <Stat label="Chunks indexed" value={(stats.chunks ?? 0).toLocaleString()} />
          <Stat label="Chunking" value={stats.chunk_strategy ?? "—"} hint={`${stats.vector_backend} backend`} />
          <Stat label="BM25 index" value={(stats.bm25_chunks ?? 0).toLocaleString()} hint="sparse arm" />
        </div>
      )}

      <Card className="mb-5 p-4">
        <div className="flex flex-wrap items-center gap-2.5">
          <input
            ref={fileInput}
            type="file"
            accept=".pdf,.txt,.docx,.md"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleUpload(file);
              e.target.value = "";
            }}
          />
          <Button variant="primary" onClick={() => fileInput.current?.click()} disabled={running}>
            Upload a contract
          </Button>
          <Button onClick={() => handleBulkIngest(50)} disabled={running}>
            Index 50 from CUAD
          </Button>
          <Button onClick={() => handleBulkIngest(510)} disabled={running}>
            Index full corpus (510)
          </Button>
          <span className="text-[12px]" style={{ color: "var(--text-muted)" }}>
            PDF, DOCX, TXT or Markdown · up to 50MB
          </span>
        </div>

        {job && (
          <div className="mt-3 rounded-lg border p-3" style={{ borderColor: "var(--border)", background: "var(--surface-sunken)" }}>
            <div className="flex items-center gap-2 text-[12.5px]">
              {running && <Spinner />}
              <span className="font-medium">
                {job.status === "completed" ? "Indexing complete" : `Indexing… ${job.processed ?? 0}/${job.total ?? "?"}`}
              </span>
              {job.status === "completed" && (
                <span style={{ color: "var(--text-muted)" }}>
                  {job.documents_indexed} documents · {(job.chunks_indexed ?? 0).toLocaleString()} chunks ·{" "}
                  {job.elapsed_s}s
                  {job.failed ? ` · ${job.failed} failed` : ""}
                  {job.skipped ? ` · ${job.skipped} already indexed` : ""}
                </span>
              )}
            </div>
            {running && job.total > 0 && (
              <div className="mt-2 h-1 overflow-hidden rounded-full" style={{ background: "var(--border)" }}>
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.round(((job.processed ?? 0) / job.total) * 100)}%`,
                    background: "var(--color-accent-500)",
                  }}
                />
              </div>
            )}
          </div>
        )}

        {error && (
          <p className="mt-2.5 text-[12.5px]" style={{ color: "var(--color-bad-500)" }}>
            {error}
          </p>
        )}
      </Card>

      <div className="mb-2.5 flex items-center gap-2.5">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search indexed documents…"
          className="w-72 rounded-lg border px-2.5 py-1.5 text-[13px] outline-none"
          style={{ borderColor: "var(--border)", background: "var(--surface)", color: "var(--text)" }}
        />
        <span className="text-[12px]" style={{ color: "var(--text-muted)" }}>
          {filtered.length} of {documents.length}
        </span>
      </div>

      {loading ? (
        <Card className="flex items-center gap-2 p-5 text-[13px]" style={{ color: "var(--text-muted)" }}>
          <Spinner /> Loading…
        </Card>
      ) : !documents.length ? (
        <Card>
          <EmptyState
            title="No documents indexed yet"
            body="Upload a contract, or index a batch from the CUAD corpus to get started."
          />
        </Card>
      ) : (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b text-left" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                  <th className="px-3.5 py-2 font-medium">Title</th>
                  <th className="px-3.5 py-2 font-medium">Type</th>
                  <th className="px-3.5 py-2 font-medium">Parties</th>
                  <th className="px-3.5 py-2 font-medium">Law</th>
                  <th className="px-3.5 py-2 text-right font-medium">Chunks</th>
                  <th className="px-3.5 py-2" />
                </tr>
              </thead>
              <tbody>
                {filtered.map((doc) => (
                  <tr key={doc.doc_id} className="border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                    <td className="max-w-[340px] px-3.5 py-2.5">
                      <span className="block truncate" title={doc.title}>
                        {doc.title}
                      </span>
                    </td>
                    <td className="px-3.5 py-2.5">
                      <Badge>{doc.contract_type ?? "Agreement"}</Badge>
                    </td>
                    <td className="max-w-[200px] px-3.5 py-2.5">
                      <span className="block truncate" style={{ color: "var(--text-muted)" }}>
                        {doc.parties?.length ? doc.parties.join(", ") : "—"}
                      </span>
                    </td>
                    <td className="px-3.5 py-2.5" style={{ color: "var(--text-muted)" }}>
                      {doc.governing_law ?? "—"}
                    </td>
                    <td className="px-3.5 py-2.5 text-right tabular-nums">{doc.chunk_count}</td>
                    <td className="px-3.5 py-2.5 text-right">
                      <button
                        onClick={() => handleDelete(doc.doc_id, doc.title)}
                        className="text-[12px] underline underline-offset-2"
                        style={{ color: "var(--text-muted)" }}
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
