"""Graph state (Module 7: State Management).

LangGraph state is a single typed dict that every node reads and updates. The
discipline it enforces is the point: a node receives state and returns a *patch*
of the keys it changed, rather than mutating shared objects. That is what makes
the pipeline inspectable - you can checkpoint it, resume it, and replay exactly
what each node saw.

The schema below is deliberately flat and JSON-serialisable so it survives the
SQLite checkpointer. Anything non-serialisable (the streaming queue, the LLM
client) travels through the graph's `config`, never through state.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

QueryIntent = Literal[
    "document_question", "summarize", "compare", "legal_advice", "out_of_scope", "meta"
]


class RAGState(TypedDict, total=False):
    # ---- request -------------------------------------------------------
    question: str
    session_id: str
    trace_id: str
    filters: dict[str, Any]
    history: list[dict[str, str]]
    history_summary: str | None

    # ---- routing (Module 7: Conditional Routing) -----------------------
    intent: QueryIntent
    needs_retrieval: bool
    is_followup: bool
    route_reason: str

    # ---- query understanding -------------------------------------------
    search_query: str
    keyword_query: str
    clause_types: list[str]

    # ---- retrieval -------------------------------------------------------
    chunks: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    context_sufficient: bool
    context_grade_reason: str

    # ---- self-correction loop -------------------------------------------
    # Bounded counters. Without a ceiling a conditional edge that routes back
    # on failure is an infinite loop that bills you for every lap.
    retrieval_attempts: int
    generation_attempts: int
    broadened: bool

    # ---- generation ------------------------------------------------------
    answer: str
    citations: list[dict[str, Any]]

    # ---- verification (Module 8) ----------------------------------------
    groundedness: dict[str, Any]
    guardrail: dict[str, Any]
    blocked: bool

    # ---- accounting ------------------------------------------------------
    # `operator.add` is the reducer: each node appends its own usage records
    # instead of overwriting the list, so the total survives the whole run.
    usage: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]

    # ---- output ----------------------------------------------------------
    final_answer: str
    warnings: list[str]


MAX_RETRIEVAL_ATTEMPTS = 2
MAX_GENERATION_ATTEMPTS = 2


def initial_state(
    question: str,
    *,
    session_id: str,
    trace_id: str,
    filters: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
    history_summary: str | None = None,
) -> RAGState:
    return RAGState(
        question=question,
        session_id=session_id,
        trace_id=trace_id,
        filters=filters or {},
        history=history or [],
        history_summary=history_summary,
        retrieval_attempts=0,
        generation_attempts=0,
        broadened=False,
        blocked=False,
        chunks=[],
        citations=[],
        warnings=[],
        usage=[],
        events=[],
    )
