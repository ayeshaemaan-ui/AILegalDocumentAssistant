"""Structured-output schemas for the graph's decision nodes (Module 2: JSON Mode).

Every conditional edge in the graph branches on one of these. That is precisely
why they are schema-constrained rather than parsed out of prose: an edge that
depends on regexing "yes" out of a sentence is a bug waiting for the model to
phrase itself differently. With `output_config.format`, the model physically
cannot return a shape the router does not understand.

The `reason` field on each is not decoration - it is what the observability
page shows when you ask why a question was routed the way it was.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Intent classification (Module 7: Conditional Routing)."""

    intent: Literal[
        "document_question", "summarize", "compare", "legal_advice", "out_of_scope", "meta"
    ] = Field(description="Which handler should take this message.")
    needs_retrieval: bool = Field(description="False only for legal_advice, out_of_scope and meta.")
    is_followup: bool = Field(description="True if it depends on earlier conversation to make sense.")
    reason: str = Field(max_length=200, description="One sentence explaining the classification.")


class QueryRewrite(BaseModel):
    """Query understanding: conversational phrasing -> contract vocabulary."""

    search_query: str = Field(description="Self-contained question in formal contract language.")
    keyword_query: str = Field(description="3-8 high-signal terms for exact-match search.")
    clause_types: list[str] = Field(
        default_factory=list,
        description="Clause-type slugs to filter on. Empty unless the mapping is obvious.",
    )


class ContextGrade(BaseModel):
    """Retrieval self-check, before spending generation tokens."""

    sufficient: bool = Field(description="Can the question be answered from these excerpts alone?")
    reason: str = Field(max_length=250, description="What is present, or what is missing.")
    missing: str = Field(
        default="", max_length=200,
        description="If insufficient, what to search for instead. Empty otherwise.",
    )


class GroundednessReport(BaseModel):
    """Hallucination detection (Module 8: Hallucination Detection)."""

    grounded_ratio: float = Field(ge=0.0, le=1.0, description="Supported claims / total claims.")
    verdict: Literal["grounded", "partially_grounded", "hallucinated"]
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims no excerpt supports, quoted verbatim."
    )
    reasoning: str = Field(max_length=400, description="Two or three sentences on the verdict.")


class AnswerGrade(BaseModel):
    """LLM-as-judge scores against a reference answer (Module 8)."""

    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    citation_quality: int = Field(ge=1, le=5)
    reasoning: str = Field(max_length=400)
