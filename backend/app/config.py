"""Central configuration.

Every knob the RAG pipeline exposes lives here so experiments (chunk size,
retrieval depth, re-ranking on/off) are reproducible from a single .env file
instead of scattered magic numbers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "LegalLens"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    # 3400 by default. 3000/3001/3002/3100 are all commonly occupied by other
    # dev servers on a working machine, so LegalLens claims a quieter port.
    cors_origins: list[str] = [
        "http://localhost:3400",
        "http://127.0.0.1:3400",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ------------------------------------------------------------- provider
    # Module 2 (LLM APIs): the app is provider-agnostic. Anthropic is primary;
    # switching to another vendor is a single env var + a new adapter class.
    llm_provider: Literal["anthropic", "openai", "google", "ollama"] = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Claude model IDs. Opus 5 answers (quality matters for legal text);
    # Haiku 4.5 handles the cheap high-volume roles (routing, grading, judging).
    answer_model: str = "claude-opus-5"
    fast_model: str = "claude-haiku-4-5"
    judge_model: str = "claude-haiku-4-5"

    # The same three roles on OpenAI. Kept as separate fields rather than
    # overwriting the Claude ones so that flipping LLM_PROVIDER back and forth
    # does not require rewriting model names each time.
    openai_answer_model: str = "gpt-5"
    openai_fast_model: str = "gpt-5-mini"
    openai_judge_model: str = "gpt-5-mini"

    # Ollama: local models, zero cost, no key. Same three roles again.
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_answer_model: str = "qwen2.5:7b-instruct"
    ollama_fast_model: str = "qwen2.5:7b-instruct"
    ollama_judge_model: str = "qwen2.5:7b-instruct"
    # A 7B model has far less context than a frontier model, so retrieved
    # excerpts are trimmed to fit rather than silently overflowing the window.
    ollama_context_tokens: int = 8192
    ollama_temperature: float = 0.2

    # Google Gemini: the free tier needs no credit card, which makes it the
    # realistic cloud option when there is no billing account.
    google_api_key: str | None = None
    # Verified available on a current free-tier key. The 2.5 line is closed to
    # new accounts, so defaulting to it would 404 for anyone starting today.
    gemini_answer_model: str = "gemini-3.6-flash"
    gemini_fast_model: str = "gemini-3.5-flash-lite"
    gemini_judge_model: str = "gemini-3.5-flash-lite"
    gemini_temperature: float = 0.2

    # Module 1 (Temperature/Top-P) + thinking/effort controls.
    answer_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    answer_max_tokens: int = 4096
    fast_max_tokens: int = 2048
    enable_prompt_cache: bool = True

    # --------------------------------------------------------- embeddings
    # Module 4: local BGE keeps the full 510-contract corpus free to re-index.
    embedding_backend: Literal["local", "anthropic_none"] = "local"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_batch_size: int = 32
    # BGE was trained with an asymmetric prefix on the query side only.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # ------------------------------------------------------- vector store
    vector_backend: Literal["chroma", "pgvector", "pinecone", "weaviate"] = "chroma"
    collection_name: str = "legallens_clauses"
    chroma_dir: Path = BACKEND_DIR / "data" / "chroma"
    pg_connection: str = "postgresql+psycopg://legallens:legallens@localhost:5432/legallens"
    pinecone_api_key: str | None = None
    pinecone_index: str = "legallens"
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str | None = None

    # ------------------------------------------------------------ chunking
    # Tier 1 M1 (Advanced Chunking). Legal text is clause-structured, so the
    # splitter is hierarchical (Article > Section > paragraph) before it ever
    # falls back to character counts.
    chunk_strategy: Literal["recursive", "legal_structural", "semantic"] = "legal_structural"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_chunk_chars: int = 120
    semantic_breakpoint_percentile: int = 90

    # ----------------------------------------------------------- retrieval
    retrieval_top_k: int = 20          # candidates from each retriever arm
    final_top_k: int = 6               # chunks actually injected into the prompt
    hybrid_enabled: bool = True        # Tier 1 M1 (Hybrid Search)
    rrf_k: int = 60                    # Reciprocal Rank Fusion constant
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rerank_enabled: bool = True        # Module 5 (Re-ranking)
    rerank_backend: Literal["cross_encoder", "llm", "none"] = "cross_encoder"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # MMR is OFF by default because the ablation said so, not by taste:
    # `scripts/ablate.py` measured it costing 0.014-0.024 nDCG in every pairing
    # on CUAD. Retrieval here is usually scoped to one contract, where
    # near-duplicate clauses are typically all relevant, so trading relevance
    # for diversity loses more than it returns. Worth re-testing for unscoped
    # search over the full corpus, where one document can flood the candidates.
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
    min_relevance_score: float = 0.0

    # ----------------------------------------------- sessions & rate limits
    session_ttl_seconds: int = 60 * 60 * 8
    max_history_turns: int = 8          # Tier 1 M2 (Chat Memory)
    summarize_history_after: int = 6    # rolling summary beyond this many turns
    rate_limit_requests: int = 30       # Tier 1 M2 (Rate Limiting)
    rate_limit_window_seconds: int = 60
    rate_limit_burst: int = 10
    daily_cost_ceiling_usd: float = 25.0  # Tier 1 M5 (Cost Tracking) circuit breaker

    # ------------------------------------------------------- observability
    # Module 8. All three are optional; the app degrades to local tracing.
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "legallens"
    helicone_api_key: str | None = None
    otel_enabled: bool = True
    otel_endpoint: str | None = None    # e.g. http://localhost:4318/v1/traces
    trace_db_path: Path = BACKEND_DIR / "data" / "traces.db"

    # ------------------------------------------------------------ storage
    upload_dir: Path = BACKEND_DIR / "data" / "uploads"
    state_db_path: Path = BACKEND_DIR / "data" / "legallens.db"
    checkpoint_db_path: Path = BACKEND_DIR / "data" / "checkpoints.db"

    # ---------------------------------------------------------- dataset
    cuad_dir: Path = Field(default=Path.home() / "Desktop" / "CUAD_v1")

    # ---------------------------------------------------------- guardrails
    # Tier 1 M3 (Agent Guardrails). Legal advice is the thing we must refuse.
    guardrails_enabled: bool = True
    max_question_chars: int = 2000
    groundedness_threshold: float = 0.6

    @property
    def resolved_answer_model(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_answer_model
        if self.llm_provider == "ollama":
            return self.ollama_answer_model
        if self.llm_provider == "google":
            return self.gemini_answer_model
        return self.answer_model

    @property
    def resolved_fast_model(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_fast_model
        if self.llm_provider == "ollama":
            return self.ollama_fast_model
        if self.llm_provider == "google":
            return self.gemini_fast_model
        return self.fast_model

    @property
    def resolved_judge_model(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_judge_model
        if self.llm_provider == "ollama":
            return self.ollama_judge_model
        if self.llm_provider == "google":
            return self.gemini_judge_model
        return self.judge_model

    def ensure_dirs(self) -> None:
        for path in (self.chroma_dir, self.upload_dir, self.state_db_path.parent):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
