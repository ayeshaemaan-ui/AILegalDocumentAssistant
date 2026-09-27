/**
 * API client.
 *
 * Requests go to relative `/api/...` paths, which next.config.mjs rewrites to
 * FastAPI. That keeps the browser on one origin - no CORS preflight, and the
 * SSE connection is same-origin, which is what makes streaming reliable behind
 * proxies.
 */
import type { AskResponse, ClauseType, DocumentSummary, EvalRun, Filters } from "./types";

const SESSION_KEY = "legallens.session";

export function getStoredSession(): string | null {
  try {
    return localStorage.getItem(SESSION_KEY);
  } catch {
    // Private browsing or blocked site data - the app works without it, the
    // conversation just will not survive a reload.
    return null;
  }
}

export function storeSession(id: string): void {
  try {
    localStorage.setItem(SESSION_KEY, id);
  } catch {
    /* non-fatal */
  }
}

export function clearStoredSession(): void {
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch {
    /* non-fatal */
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const sessionId = typeof window !== "undefined" ? getStoredSession() : null;
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(sessionId ? { "X-Session-Id": sessionId } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* keep the status line */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = "ApiError";
  }
}

export interface StreamHandlers {
  onMeta?: (data: { session_id: string; trace_id: string }) => void;
  onToken?: (text: string) => void;
  onThinking?: (text: string) => void;
  onRestart?: (reason: string) => void;
  onDone?: (data: AskResponse) => void;
  onError?: (data: { kind: string; message: string; retry_after?: number }) => void;
}

/**
 * Stream an answer over Server-Sent Events.
 *
 * Uses fetch + a ReadableStream rather than EventSource because EventSource
 * cannot issue a POST, and the question plus its filters belong in a body, not
 * a query string.
 */
export async function streamAsk(
  question: string,
  filters: Filters,
  sessionId: string | null,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(sessionId ? { "X-Session-Id": sessionId } : {}),
    },
    body: JSON.stringify({ question, session_id: sessionId, ...filters }),
    signal,
  });

  if (!response.ok || !response.body) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) message = String(body.detail);
    } catch {
      /* keep the status line */
    }
    handlers.onError?.({ kind: "http", message });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. A partial frame stays in the
    // buffer until its terminator arrives.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      let event = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;

      let payload: any;
      try {
        payload = JSON.parse(dataLines.join("\n"));
      } catch {
        continue;
      }

      switch (event) {
        case "meta": handlers.onMeta?.(payload); break;
        case "token": handlers.onToken?.(payload.text ?? ""); break;
        case "thinking": handlers.onThinking?.(payload.text ?? ""); break;
        case "restart": handlers.onRestart?.(payload.reason ?? "regenerating"); break;
        case "done": handlers.onDone?.(payload); break;
        case "error": handlers.onError?.(payload); break;
      }
    }
  }
}

export const api = {
  ask: (question: string, filters: Filters, sessionId: string | null) =>
    request<AskResponse>("/api/chat/ask", {
      method: "POST",
      body: JSON.stringify({ question, session_id: sessionId, ...filters }),
    }),

  runAgent: (question: string, mode: "agent" | "orchestrator", sessionId: string | null) =>
    request<any>("/api/chat/agent", {
      method: "POST",
      body: JSON.stringify({ question, mode, session_id: sessionId }),
    }),

  documents: () => request<DocumentSummary[]>("/api/documents"),
  documentStats: () => request<any>("/api/documents/stats"),
  clauseTypes: () => request<{ clause_types: ClauseType[] }>("/api/documents/clause-types"),
  deleteDocument: (docId: string) =>
    request<any>(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" }),

  ingest: (limit: number, strategy?: string) =>
    request<{ job_id: string; total: number }>("/api/documents/ingest", {
      method: "POST",
      body: JSON.stringify({ limit, strategy }),
    }),
  ingestJob: (jobId: string) => request<any>(`/api/documents/jobs/${jobId}`),

  upload: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/api/documents/upload", { method: "POST", body: form });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new ApiError(body?.detail ?? `Upload failed (${response.status})`, response.status);
    }
    return response.json();
  },

  health: () => request<any>("/api/health"),
  config: () => request<any>("/api/config"),
  graph: () => request<{ mermaid: string }>("/api/graph"),
  costSummary: (days = 7) => request<any>(`/api/cost/summary?days=${days}`),
  traces: (limit = 50) => request<any>(`/api/traces?limit=${limit}`),
  trace: (traceId: string) => request<any>(`/api/traces/${traceId}`),
  sessions: () => request<any>("/api/sessions"),
  deleteSession: (id: string) => request<any>(`/api/sessions/${id}`, { method: "DELETE" }),

  // ---- clause matrix extraction ----
  startMatrix: (body: { doc_ids: string[]; clause_slugs: string[]; concurrency?: number }) =>
    request<any>("/api/extract/matrix", { method: "POST", body: JSON.stringify(body) }),
  matrixJob: (jobId: string) => request<any>(`/api/extract/jobs/${jobId}`),
  matrixCsvUrl: (jobId: string) => `/api/extract/jobs/${jobId}/csv`,

  evalRuns: () => request<{ runs: EvalRun[] }>("/api/eval/runs"),
  evalRun: (runId: string) => request<any>(`/api/eval/runs/${runId}`),
  goldenSet: () => request<any>("/api/eval/golden-set"),
  startEval: (body: { sample_size: number; retrieval_only: boolean; judge: boolean; seed?: number }) =>
    request<any>("/api/eval/run", { method: "POST", body: JSON.stringify(body) }),
  evalJob: (jobId: string) => request<any>(`/api/eval/jobs/${jobId}`),
  compareRuns: (baseline: string, candidate: string) =>
    request<any>(`/api/eval/compare?baseline=${baseline}&candidate=${candidate}`),
};
