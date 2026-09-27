"""Graph nodes (Module 7: Nodes).

Each function here is one step of the pipeline. They share a strict contract:
take the state, do one job, return a *patch* of the keys they changed. No node
mutates state in place, and no node raises into the graph - a failure becomes a
recorded event and a degraded-but-valid patch, because a research tool that
returns a partial answer is better than one that 500s.

Read in order, they are the whole system:

    guard_input -> route -> rewrite_query -> retrieve -> grade_context
                -> generate -> verify -> finalize

with `broaden` as the retry arm and `refuse` as the early exit.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.config import get_settings
from app.core import guardrails
from app.core.cost import get_cost_tracker
from app.core.llm import get_llm_client
from app.graph.schemas import ContextGrade, GroundednessReport, QueryRewrite, RouteDecision
from app.graph.state import (
    MAX_GENERATION_ATTEMPTS,
    MAX_RETRIEVAL_ATTEMPTS,
    RAGState,
)
from app.ingestion.taxonomy import BY_SLUG
from app.prompts import templates
from app.retrieval.retriever import RetrievalFilters, get_retriever
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- utilities
def _event(name: str, **payload: Any) -> dict[str, Any]:
    return {"node": name, "ts": time.time(), **payload}


def _record(state: RAGState, usage, stage: str) -> dict[str, Any]:
    """Log one model call to the cost ledger and return it for the state patch."""
    tracker = get_cost_tracker()
    tracker.record(usage, stage=stage, session_id=state.get("session_id"), trace_id=state.get("trace_id"))
    return {**usage.as_dict(), "stage": stage}


def _chunks_from_state(state: RAGState) -> list[RetrievedChunk]:
    """Rehydrate chunk dicts back into objects for prompting and validation."""
    out: list[RetrievedChunk] = []
    for payload in state.get("chunks", []):
        chunk = RetrievedChunk(
            chunk_id=payload.get("chunk_id", ""),
            doc_id=payload.get("doc_id", ""),
            text=payload.get("text", ""),
            metadata=payload.get("metadata", {}),
        )
        chunk.final_score = (payload.get("scores") or {}).get("final") or 0.0
        out.append(chunk)
    return out


# =========================================================== 1. INPUT GUARD
async def guard_input(state: RAGState) -> dict[str, Any]:
    """Cheap deterministic checks before anything costs money."""
    verdict = guardrails.check_input(state["question"])
    patch: dict[str, Any] = {
        "guardrail": verdict.as_dict(),
        "warnings": list(verdict.warnings),
        "events": [_event("guard_input", allowed=verdict.allowed, category=verdict.category)],
    }
    if not verdict.allowed:
        patch["blocked"] = True
        patch["final_answer"] = verdict.replacement or guardrails.build_refusal("out_of_scope")
        patch["answer"] = patch["final_answer"]
    return patch


# ================================================================= 2. ROUTE
async def route(state: RAGState) -> dict[str, Any]:
    """Classify intent with a small, cheap model (Module 7: Conditional Routing)."""
    settings = get_settings()
    client = get_llm_client()

    context = ""
    if state.get("history_summary"):
        context = f"Conversation so far:\n{state['history_summary']}\n\n"

    try:
        decision, usage = await client.structured(
            system=templates.ROUTER_SYSTEM,
            messages=[{"role": "user", "content": f"{context}Message: {state['question']}"}],
            schema=RouteDecision,
            model=settings.resolved_fast_model,
            effort="low",
        )
    except Exception as exc:
        # Routing must never be the thing that breaks a question. Default to
        # the common case and carry on.
        logger.warning("Router failed (%s); defaulting to document_question", exc)
        return {
            "intent": "document_question",
            "needs_retrieval": True,
            "is_followup": False,
            "route_reason": f"router unavailable ({type(exc).__name__})",
            "events": [_event("route", fallback=True, error=str(exc))],
        }

    return {
        "intent": decision.intent,
        "needs_retrieval": decision.needs_retrieval,
        "is_followup": decision.is_followup,
        "route_reason": decision.reason,
        "usage": [_record(state, usage, "graph.route")],
        "events": [_event("route", intent=decision.intent, reason=decision.reason)],
    }


# ========================================================= 3. QUERY REWRITE
async def rewrite_query(state: RAGState) -> dict[str, Any]:
    """Bridge user phrasing to contract vocabulary, and resolve follow-ups."""
    settings = get_settings()
    client = get_llm_client()
    question = state["question"]

    try:
        rewrite, usage = await client.structured(
            system=templates.QUERY_REWRITE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": templates.build_query_rewrite_turn(
                        question, list(BY_SLUG.keys()), state.get("history_summary")
                    ),
                }
            ],
            schema=QueryRewrite,
            model=settings.resolved_fast_model,
            effort="low",
        )
    except Exception as exc:
        logger.warning("Query rewrite failed (%s); using the raw question", exc)
        return {
            "search_query": question,
            "keyword_query": question,
            "clause_types": [],
            "events": [_event("rewrite_query", fallback=True, error=str(exc))],
        }

    # The model can only pick from the taxonomy; drop anything invented.
    slugs = [s for s in rewrite.clause_types if s in BY_SLUG]
    return {
        "search_query": rewrite.search_query or question,
        "keyword_query": rewrite.keyword_query or question,
        "clause_types": slugs,
        "usage": [_record(state, usage, "graph.rewrite")],
        "events": [
            _event(
                "rewrite_query",
                search_query=rewrite.search_query,
                keyword_query=rewrite.keyword_query,
                clause_types=slugs,
            )
        ],
    }


# ============================================================== 4. RETRIEVE
async def retrieve(state: RAGState) -> dict[str, Any]:
    """Hybrid retrieval + re-ranking, with the session's filters applied."""
    settings = get_settings()
    retriever = get_retriever()
    session_filters = state.get("filters") or {}

    # Clause filters come from the rewriter, but only on the first attempt -
    # a broadened retry deliberately drops them.
    clause_types = [] if state.get("broadened") else state.get("clause_types", [])

    filters = RetrievalFilters(
        doc_ids=session_filters.get("doc_ids", []),
        clause_types=clause_types,
        contract_types=session_filters.get("contract_types", []),
        governing_law=session_filters.get("governing_law"),
    )

    # Dense search reads the rewritten question; BM25 gets the keyword form
    # appended so exact terms are weighted without polluting the embedding.
    dense_query = state.get("search_query") or state["question"]
    keyword_query = state.get("keyword_query") or ""
    query = f"{dense_query} {keyword_query}".strip() if keyword_query else dense_query

    # `summarize` and `compare` need a wider view than a point lookup.
    top_k = settings.final_top_k
    if state.get("intent") in ("summarize", "compare"):
        top_k = max(top_k * 2, 12)

    try:
        chunks, trace = await retriever.retrieve(query, filters=filters, top_k=top_k)
    except Exception as exc:
        logger.exception("Retrieval failed")
        return {
            "chunks": [],
            "retrieval_trace": {"error": str(exc)},
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            "events": [_event("retrieve", error=str(exc))],
        }

    return {
        "chunks": [c.to_dict() for c in chunks],
        "retrieval_trace": trace.as_dict(),
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "events": [
            _event(
                "retrieve",
                returned=len(chunks),
                dense=trace.dense_count,
                sparse=trace.sparse_count,
                reranked=trace.reranked,
            )
        ],
    }


# ========================================================= 5. GRADE CONTEXT
async def grade_context(state: RAGState) -> dict[str, Any]:
    """Ask whether the retrieved excerpts can actually answer, before generating.

    This is the cheap half of self-correction: a Haiku call that decides whether
    to spend an Opus call. When it says no, the graph broadens the search rather
    than generating a confident answer from the wrong clauses.
    """
    settings = get_settings()
    chunks = state.get("chunks", [])
    if not chunks:
        return {
            "context_sufficient": False,
            "context_grade_reason": "No excerpts retrieved.",
            "events": [_event("grade_context", sufficient=False, reason="empty")],
        }

    # On the final attempt, grading changes nothing - skip the call and answer
    # with what we have.
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return {
            "context_sufficient": True,
            "context_grade_reason": "Retrieval budget exhausted; answering from best available.",
            "events": [_event("grade_context", sufficient=True, reason="budget_exhausted")],
        }

    excerpts = "\n\n".join(
        f"[{i}] {c['metadata'].get('section_path') or c['metadata'].get('title', '')}\n{c['text'][:900]}"
        for i, c in enumerate(chunks, start=1)
    )
    system = (
        "You decide whether a set of contract excerpts is sufficient to answer a "
        "question. Be strict about the distinction between related and sufficient: "
        "excerpts on the right topic that never state the specific term asked about "
        "are insufficient. But excerpts that clearly show the contract is silent on "
        "the issue ARE sufficient - reporting an absence is a valid answer."
    )
    client = get_llm_client()
    try:
        grade, usage = await client.structured(
            system=system,
            messages=[{"role": "user", "content": f"Question: {state['question']}\n\nExcerpts:\n{excerpts}"}],
            schema=ContextGrade,
            model=settings.resolved_fast_model,
            effort="low",
        )
    except Exception as exc:
        logger.warning("Context grading failed (%s); proceeding to generate", exc)
        return {
            "context_sufficient": True,
            "context_grade_reason": "grader unavailable",
            "events": [_event("grade_context", fallback=True, error=str(exc))],
        }

    return {
        "context_sufficient": grade.sufficient,
        "context_grade_reason": grade.reason,
        "search_query": grade.missing or state.get("search_query", ""),
        "usage": [_record(state, usage, "graph.grade")],
        "events": [_event("grade_context", sufficient=grade.sufficient, reason=grade.reason)],
    }


# =============================================================== 6. BROADEN
async def broaden(state: RAGState) -> dict[str, Any]:
    """Relax the search and try again (Module 7: Multi-Step Workflows).

    Drops clause-type filters and widens the candidate pool. This is the loop
    edge of the graph, and `MAX_RETRIEVAL_ATTEMPTS` is what keeps it a loop
    rather than a spiral.
    """
    return {
        "broadened": True,
        "clause_types": [],
        "events": [
            _event("broaden", reason=state.get("context_grade_reason", ""), attempt=state.get("retrieval_attempts", 0))
        ],
    }


# ============================================================== 7. GENERATE
async def generate(state: RAGState, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate the grounded answer (Module 5: Context Injection).

    Streams tokens into the queue supplied via `config` when one is present, so
    the same node serves both the live UI and the batch evaluation runner.
    """
    settings = get_settings()
    client = get_llm_client()
    chunks = _chunks_from_state(state)

    if not chunks:
        answer = templates.NO_RESULTS_RESPONSE
        queue = (config or {}).get("configurable", {}).get("stream_queue")
        if queue is not None:
            await queue.put({"type": "token", "text": answer})
        return {
            "answer": answer,
            "final_answer": answer,
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "events": [_event("generate", skipped="no_chunks")],
        }

    intent = state.get("intent", "document_question")
    system = {
        "summarize": templates.SUMMARY_SYSTEM,
        "compare": templates.COMPARE_SYSTEM,
    }.get(intent, templates.ANSWER_SYSTEM)

    user_turn = templates.build_answer_user_turn(
        state["question"], chunks, history_summary=state.get("history_summary")
    )

    # A retry after a failed groundedness check gets a sharper instruction,
    # appended to the *user* turn so the cached system prefix stays byte-stable.
    attempt = state.get("generation_attempts", 0)
    if attempt > 0:
        unsupported = (state.get("groundedness") or {}).get("unsupported_claims", [])
        user_turn += (
            "\n\n## Correction required\n"
            "Your previous answer contained claims the excerpts do not support:\n"
            + "\n".join(f"- {c}" for c in unsupported[:5])
            + "\nRewrite it using only what the excerpts state. Where the excerpts are "
            "silent, say so explicitly instead of inferring."
        )

    messages = list(state.get("history", []))
    messages.append({"role": "user", "content": user_turn})

    queue = (config or {}).get("configurable", {}).get("stream_queue")
    parts: list[str] = []
    usage_payload: dict[str, Any] | None = None

    try:
        if queue is not None:
            async for event in client.stream(
                system=system,
                messages=messages,
                model=settings.resolved_answer_model,
                max_tokens=settings.answer_max_tokens,
                effort=settings.answer_effort,
            ):
                if event["type"] == "token":
                    parts.append(event["text"])
                    await queue.put(event)
                elif event["type"] == "thinking":
                    await queue.put(event)
                elif event["type"] == "usage":
                    usage_payload = {**event["usage"], "stage": "graph.generate"}
            answer = "".join(parts).strip()
        else:
            result = await client.complete(
                system=system,
                messages=messages,
                model=settings.resolved_answer_model,
                max_tokens=settings.answer_max_tokens,
                effort=settings.answer_effort,
            )
            answer = result.text
            usage_payload = _record(state, result.usage, "graph.generate")
    except Exception as exc:
        logger.exception("Generation failed")
        message = _generation_error_message(exc)
        if queue is not None:
            await queue.put({"type": "token", "text": message})
        return {
            "answer": message,
            "final_answer": message,
            "generation_attempts": attempt + 1,
            "events": [_event("generate", error=str(exc))],
        }

    if usage_payload and queue is not None:
        # Streaming bypasses `_record`, so bill it here.
        from app.core.llm import Usage

        get_cost_tracker().record(
            Usage(
                model=usage_payload.get("model", settings.resolved_answer_model),
                input_tokens=usage_payload.get("input_tokens", 0),
                output_tokens=usage_payload.get("output_tokens", 0),
                cache_read_tokens=usage_payload.get("cache_read_tokens", 0),
                cache_write_tokens=usage_payload.get("cache_write_tokens", 0),
                latency_ms=usage_payload.get("latency_ms", 0.0),
            ),
            stage="graph.generate",
            session_id=state.get("session_id"),
            trace_id=state.get("trace_id"),
        )

    return {
        "answer": answer,
        "generation_attempts": attempt + 1,
        "usage": [usage_payload] if usage_payload else [],
        "events": [_event("generate", chars=len(answer), attempt=attempt + 1)],
    }


# ================================================================ 8. VERIFY
async def verify(state: RAGState) -> dict[str, Any]:
    """Validate citations, then audit groundedness (Module 8).

    Two checks in increasing order of cost. Citation validation is a regex and
    catches fabricated evidence for free. Only if that passes do we spend a
    model call on the harder question of whether the prose is supported.
    """
    settings = get_settings()
    chunks = _chunks_from_state(state)
    answer = state.get("answer", "")

    verdict = guardrails.check_output(answer, chunks)
    warnings = list(state.get("warnings", [])) + verdict.warnings

    if not verdict.allowed:
        return {
            "guardrail": verdict.as_dict(),
            "warnings": warnings,
            "groundedness": {
                "verdict": "hallucinated",
                "grounded_ratio": 0.0,
                "reasoning": verdict.reason or "Invalid citations.",
                "unsupported_claims": [],
            },
            "events": [_event("verify", blocked=True, reason=verdict.reason)],
        }

    if not chunks or not answer.strip():
        return {
            "guardrail": verdict.as_dict(),
            "warnings": warnings,
            "groundedness": {"verdict": "grounded", "grounded_ratio": 1.0, "reasoning": "No claims made."},
            "events": [_event("verify", skipped="no_content")],
        }

    client = get_llm_client()
    try:
        report, usage = await client.structured(
            system=templates.GROUNDEDNESS_SYSTEM,
            messages=[
                {"role": "user", "content": templates.build_groundedness_turn(state["question"], answer, chunks)}
            ],
            schema=GroundednessReport,
            model=settings.resolved_judge_model,
            effort="low",
            max_tokens=1024,
        )
    except Exception as exc:
        logger.warning("Groundedness check failed (%s)", exc)
        return {
            "guardrail": verdict.as_dict(),
            "warnings": warnings + ["groundedness_check_unavailable"],
            "groundedness": {"verdict": "unknown", "grounded_ratio": -1.0, "reasoning": str(exc)},
            "events": [_event("verify", error=str(exc))],
        }

    if report.verdict != "grounded":
        warnings.append(f"groundedness:{report.verdict}")

    return {
        "guardrail": verdict.as_dict(),
        "warnings": warnings,
        "groundedness": report.model_dump(),
        "usage": [_record(state, usage, "graph.verify")],
        "events": [_event("verify", verdict=report.verdict, ratio=report.grounded_ratio)],
    }


# ================================================================ 9. REFUSE
async def refuse(state: RAGState) -> dict[str, Any]:
    """Early exit for advice-seeking and out-of-scope messages."""
    if state.get("blocked") and state.get("final_answer"):
        return {"events": [_event("refuse", category="input_guardrail")]}

    intent = state.get("intent", "out_of_scope")
    answer = guardrails.build_refusal("legal_advice" if intent == "legal_advice" else "out_of_scope")
    return {
        "answer": answer,
        "final_answer": answer,
        "blocked": True,
        "events": [_event("refuse", category=intent)],
    }


# ================================================================= 10. META
async def answer_meta(state: RAGState) -> dict[str, Any]:
    """Questions about the corpus itself - answered from the manifest, not the model."""
    from app.ingestion.pipeline import get_pipeline

    pipeline = get_pipeline()
    docs = pipeline.manifest.list()
    stats = pipeline.stats()

    if not docs:
        answer = (
            "No documents are indexed yet. Upload a contract (PDF, DOCX or TXT) and "
            "I'll index it, then you can ask questions about it."
        )
    else:
        preview = "\n".join(
            f"- {d['title'][:80]} ({d.get('contract_type') or 'Agreement'}, {d['chunk_count']} chunks)"
            for d in docs[:10]
        )
        more = f"\n…and {len(docs) - 10} more." if len(docs) > 10 else ""
        answer = (
            f"I have {stats['documents']} document(s) indexed as {stats['chunks']} searchable "
            f"chunks, using the `{stats['chunk_strategy']}` chunking strategy and "
            f"`{stats['embedding_model']}` embeddings.\n\n{preview}{more}"
        )

    return {
        "answer": answer,
        "final_answer": answer,
        "events": [_event("answer_meta", documents=len(docs))],
    }


# ============================================================= 11. FINALIZE
async def finalize(state: RAGState) -> dict[str, Any]:
    """Assemble citations and attach any confidence warning for the UI."""
    chunks = state.get("chunks", [])
    answer = state.get("answer", "")

    valid, _invalid, _coverage = guardrails.validate_citations(answer, _chunks_from_state(state))
    cited = set(valid)

    citations: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        citations.append(
            {
                "index": idx,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "title": meta.get("title", ""),
                "section_path": meta.get("section_path"),
                "page_number": meta.get("page_number"),
                "page_confidence": meta.get("page_confidence", "estimated"),
                "clause_types": [s for s in (meta.get("clause_types_csv") or "").split(",") if s],
                "score": (chunk.get("scores") or {}).get("final"),
                "retriever": chunk.get("retriever"),
                "text": chunk.get("text", ""),
                "used": idx in cited,
            }
        )

    grounding = state.get("groundedness") or {}
    warnings = list(state.get("warnings", []))
    final = answer

    # Surface low confidence in the answer itself rather than hiding it in a
    # field nobody reads.
    if grounding.get("verdict") == "partially_grounded":
        final += (
            "\n\n---\n*Confidence note: parts of this answer could not be matched to the "
            "retrieved excerpts. Please verify against the cited clauses.*"
        )
    elif "no_citations" in warnings and cited == set():
        final += (
            "\n\n---\n*Confidence note: this answer does not cite a specific clause. "
            "Treat it as a starting point rather than a finding.*"
        )

    return {
        "citations": citations,
        "final_answer": final,
        "warnings": warnings,
        "events": [_event("finalize", citations=len(citations), used=len(cited))],
    }


def _generation_error_message(exc: Exception) -> str:
    """Turn a provider exception into something the reader can act on.

    Telling someone to retry a request that can never succeed - an exhausted
    billing account, a revoked key - wastes their time and hides the real
    problem. So terminal failures are named as terminal, and only genuinely
    transient ones suggest retrying. Retrieval succeeded in every case, so the
    excerpts below the message are still worth reading.
    """
    text = str(exc).lower()
    retrieved_note = (
        " The relevant excerpts were retrieved successfully and are listed below, "
        "so you can still read the source clauses."
    )

    if "no credits" in text or "insufficient_quota" in text or "billing" in text:
        return (
            "**The language model account has no credits remaining**, so the answer "
            "could not be generated. Retrying will not help until credits are added "
            "to the configured provider account." + retrieved_note
        )
    if "authentication" in text or "invalid api key" in text or "401" in text:
        return (
            "**The API key was rejected by the provider.** Check `LLM_PROVIDER` and the "
            "matching key in `.env`, then restart the backend." + retrieved_note
        )
    if "rate limit" in text or "429" in text:
        return (
            "**Rate limited by the model provider.** This one is transient - waiting a "
            "few seconds and asking again usually works." + retrieved_note
        )
    if "timeout" in text or "timed out" in text:
        return (
            "**The model call timed out.** Retrying usually works; if it persists, the "
            "question may be producing an unusually long answer." + retrieved_note
        )
    return (
        "I hit an error generating the answer." + retrieved_note + " Retrying is worth "
        "a try; the backend log has the specific error."
    )

