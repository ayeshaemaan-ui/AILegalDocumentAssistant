// Mirrors the Pydantic models in backend/app/api/schemas.py.

export interface Citation {
  index: number;
  chunk_id?: string | null;
  doc_id?: string | null;
  title: string;
  section_path?: string | null;
  page_number?: number | null;
  page_confidence: "exact" | "recovered" | "estimated";
  clause_types: string[];
  score?: number | null;
  retriever?: string | null;
  text: string;
  used: boolean;
}

export interface RetrievalTrace {
  query?: string;
  dense_count?: number;
  sparse_count?: number;
  fused_count?: number;
  after_mmr?: number;
  final_count?: number;
  reranked?: boolean;
  filters_applied?: Record<string, unknown> | null;
  timings_ms?: Record<string, number>;
  fallback_used?: string | null;
}

export interface Groundedness {
  verdict?: "grounded" | "partially_grounded" | "hallucinated" | "unknown";
  grounded_ratio?: number;
  unsupported_claims?: string[];
  reasoning?: string;
}

export interface Usage {
  calls?: number;
  total_tokens?: number;
  cost_usd?: number;
  by_stage?: Array<Record<string, unknown>>;
}

export interface AskResponse {
  session_id: string;
  trace_id: string;
  answer: string;
  citations: Citation[];
  intent?: string | null;
  route_reason?: string | null;
  search_query?: string | null;
  clause_types: string[];
  retrieval: RetrievalTrace;
  groundedness: Groundedness;
  guardrail: Record<string, unknown>;
  warnings: string[];
  blocked: boolean;
  elapsed_ms: number;
  usage: Usage;
  streamed?: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  thinking?: string;
  meta?: AskResponse;
  error?: string;
}

export interface DocumentSummary {
  doc_id: string;
  title: string;
  contract_type?: string | null;
  governing_law?: string | null;
  parties: string[];
  file_type?: string | null;
  page_count?: number | null;
  chunk_count: number;
  chunk_strategy?: string | null;
  indexed_at?: number | null;
}

export interface ClauseType {
  slug: string;
  label: string;
  description: string;
}

export interface Filters {
  doc_ids: string[];
  clause_types: string[];
  contract_types: string[];
  governing_law: string | null;
}

export interface EvalRun {
  run_id: string;
  created_at: number;
  mode: string;
  sample_size: number;
  elapsed_s: number;
  cost_usd: number;
  config: Record<string, unknown>;
  metrics: {
    retrieval?: Record<string, { mean: number; p50: number; p10: number; p90: number; n: number }>;
    answer_quality?: Record<string, number>;
    faithfulness?: { verdicts?: Record<string, number>; hallucination_rate?: number; mean_grounded_ratio?: number };
    citations?: Record<string, number>;
    latency_ms?: { mean: number; p50: number; p90: number };
    by_category_ndcg?: Record<string, number>;
    weakest_categories?: Array<{ category: string; ndcg: number; n: number }>;
    errors?: number;
    scored?: number;
  };
}
