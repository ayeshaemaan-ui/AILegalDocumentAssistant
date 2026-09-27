"""API request/response models.

Pydantic at the edge does two jobs: it validates what comes in (so a malformed
request fails at the boundary with a useful 422 rather than deep inside the
graph), and it documents the API - FastAPI generates the OpenAPI schema from
these, which is what the frontend's types are written against.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    doc_ids: list[str] = Field(default_factory=list)
    clause_types: list[str] = Field(default_factory=list)
    contract_types: list[str] = Field(default_factory=list)
    governing_law: str | None = None

    def filters(self) -> dict[str, Any]:
        return {
            "doc_ids": self.doc_ids,
            "clause_types": self.clause_types,
            "contract_types": self.contract_types,
            "governing_law": self.governing_law,
        }


class Citation(BaseModel):
    index: int
    chunk_id: str | None = None
    doc_id: str | None = None
    title: str = ""
    section_path: str | None = None
    page_number: int | None = None
    page_confidence: str = "estimated"
    clause_types: list[str] = Field(default_factory=list)
    score: float | None = None
    retriever: str | None = None
    text: str = ""
    used: bool = False


class AskResponse(BaseModel):
    session_id: str
    trace_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    intent: str | None = None
    route_reason: str | None = None
    search_query: str | None = None
    clause_types: list[str] = Field(default_factory=list)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    groundedness: dict[str, Any] = Field(default_factory=dict)
    guardrail: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    blocked: bool = False
    elapsed_ms: float = 0.0
    usage: dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    mode: Literal["agent", "orchestrator"] = "agent"
    max_steps: int = Field(default=8, ge=1, le=15)


class IngestRequest(BaseModel):
    """Bulk-index from a server-side directory (used for the CUAD corpus)."""

    directory: str | None = None
    limit: int = Field(default=50, ge=1, le=600)
    strategy: Literal["recursive", "legal_structural", "semantic"] | None = None
    force: bool = False


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    contract_type: str | None = None
    governing_law: str | None = None
    parties: list[str] = Field(default_factory=list)
    file_type: str | None = None
    page_count: int | None = None
    chunk_count: int = 0
    chunk_strategy: str | None = None
    indexed_at: float | None = None


class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class EvalRequest(BaseModel):
    sample_size: int = Field(default=25, ge=1, le=300)
    seed: int = 42
    categories: list[str] = Field(default_factory=list)
    judge: bool = Field(default=True, description="Run the LLM-as-judge answer grading.")
    retrieval_only: bool = Field(
        default=False,
        description="Score retrieval only - no generation, so no generation cost.",
    )
