"""Graph assembly (Module 7: Nodes, Edges, Conditional Routing, Multi-Step Workflows).

The shape of the pipeline, drawn in code:

                        START
                          |
                     guard_input
                     /         \\
               (blocked)      (ok)
                   |             |
                refuse         route ------------------.
                   |          /      \\                 \\
                  END   (advice/  (document/          (meta)
                         out of    summarize/            |
                         scope)    compare)          answer_meta
                            |         |                  |
                         refuse  rewrite_query           END
                            |         |
                           END    retrieve <------.
                                      |            |
                                 grade_context     |
                                  /        \\       |
                            (enough)   (not enough, |
                                 |      retries left)
                                 |          \\      |
                                 |          broaden-'
                                 |
                              generate <-------.
                                 |              |
                              verify            |
                              /     \\           |
                       (grounded) (hallucinated,|
                             |      retry left) |
                             |          \\-------'
                          finalize
                             |
                            END

Two cycles make this a workflow rather than a chain, and both are bounded:

  retrieve -> grade -> broaden -> retrieve   recovers from a bad first search
  generate -> verify -> generate             recovers from an ungrounded answer

That self-correction is the practical difference between a demo and something
you would trust. A plain chain generates from whatever it retrieved and reports
whatever it generated; this one checks its own work at both stages and gets a
second attempt, with a hard ceiling so a persistent failure ends in a degraded
answer rather than an unbounded bill.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import (
    MAX_GENERATION_ATTEMPTS,
    MAX_RETRIEVAL_ATTEMPTS,
    RAGState,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------- conditional edges
def after_guard(state: RAGState) -> Literal["refuse", "route"]:
    return "refuse" if state.get("blocked") else "route"


def after_route(state: RAGState) -> Literal["refuse", "answer_meta", "rewrite_query"]:
    intent = state.get("intent", "document_question")
    if intent in ("legal_advice", "out_of_scope"):
        return "refuse"
    if intent == "meta":
        return "answer_meta"
    return "rewrite_query"


def after_grade(state: RAGState) -> Literal["broaden", "generate"]:
    """Retry retrieval only while there is budget and we have not already broadened."""
    if state.get("context_sufficient", True):
        return "generate"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS or state.get("broadened"):
        # Out of retries: answer from the best available excerpts. `finalize`
        # will attach a confidence note.
        return "generate"
    return "broaden"


def after_verify(state: RAGState) -> Literal["generate", "finalize"]:
    """Regenerate once if the answer was not grounded in its own sources."""
    grounding = state.get("groundedness") or {}
    if (
        grounding.get("verdict") == "hallucinated"
        and state.get("generation_attempts", 0) < MAX_GENERATION_ATTEMPTS
    ):
        logger.info("Ungrounded answer - regenerating with correction instruction")
        return "generate"
    return "finalize"


# ------------------------------------------------------------ construction
def build_graph(checkpointer: Any | None = None):
    """Compile the RAG workflow.

    `checkpointer` persists state between steps, which is what lets a long run
    be resumed or inspected after the fact. Defaults to in-memory; pass an
    AsyncSqliteSaver for durability across restarts.
    """
    graph = StateGraph(RAGState)

    graph.add_node("guard_input", nodes.guard_input)
    graph.add_node("route", nodes.route)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_context", nodes.grade_context)
    graph.add_node("broaden", nodes.broaden)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify", nodes.verify)
    graph.add_node("refuse", nodes.refuse)
    graph.add_node("answer_meta", nodes.answer_meta)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", after_guard, {"refuse": "refuse", "route": "route"})
    graph.add_conditional_edges(
        "route",
        after_route,
        {"refuse": "refuse", "answer_meta": "answer_meta", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_context")
    graph.add_conditional_edges(
        "grade_context", after_grade, {"broaden": "broaden", "generate": "generate"}
    )
    graph.add_edge("broaden", "retrieve")          # retrieval retry cycle
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify", after_verify, {"generate": "generate", "finalize": "finalize"}
    )                                              # generation retry cycle
    graph.add_edge("finalize", END)
    graph.add_edge("refuse", END)
    graph.add_edge("answer_meta", END)

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


_compiled = None


def get_graph():
    """Compiled graph singleton - compilation is not free, and it is stateless."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
        logger.info("LangGraph RAG workflow compiled")
    return _compiled


def reset_graph() -> None:
    global _compiled
    _compiled = None


def render_mermaid() -> str:
    """Mermaid source for the graph - rendered on the UI's architecture page."""
    try:
        return get_graph().get_graph().draw_mermaid()
    except Exception as exc:  # pragma: no cover
        logger.warning("Mermaid render failed: %s", exc)
        return ""
