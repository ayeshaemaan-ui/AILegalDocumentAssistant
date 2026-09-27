"""Tools (Tier 1 M3: Tool Calling / Function Calling + Module 6: Tools).

The RAG graph answers one question against one retrieval. Some questions are
not that shape: "compare the termination rights across all three distributor
agreements" needs several searches whose parameters depend on what earlier
searches found. That is what tools are for - the model decides which searches
to run instead of the pipeline deciding in advance.

Each tool is defined once and exposed twice:
  * as an Anthropic tool schema for the native tool-calling loop, with
    `strict: True` so arguments are guaranteed to validate against the schema;
  * as a LangChain `StructuredTool`, so the same function works inside a
    LangChain agent or LCEL chain without a second implementation.

Guardrail note: every tool is read-only. There is no tool that writes, deletes,
or calls out to the network. An agent loop can be wrong; it should not be able
to be destructive.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.taxonomy import BY_SLUG, category_labels
from app.retrieval.retriever import RetrievalFilters, get_retriever

logger = logging.getLogger(__name__)


# ================================================================== SCHEMAS
class SearchContractsArgs(BaseModel):
    query: str = Field(description="What to search for, in contract vocabulary.")
    clause_type: str | None = Field(
        default=None, description="Optional clause-type slug to filter on (see list_clause_types)."
    )
    doc_id: str | None = Field(default=None, description="Optional: restrict to one document.")
    top_k: int = Field(default=5, ge=1, le=15, description="How many excerpts to return.")


class DocumentArgs(BaseModel):
    doc_id: str = Field(description="The document identifier, from list_documents.")


class CompareClauseArgs(BaseModel):
    clause_type: str = Field(description="Clause-type slug to compare across documents.")
    doc_ids: list[str] = Field(description="Two or more document ids to compare.")


# ============================================================ IMPLEMENTATIONS
async def search_contracts(
    query: str, clause_type: str | None = None, doc_id: str | None = None, top_k: int = 5
) -> str:
    """Hybrid search over the indexed contracts."""
    retriever = get_retriever()
    filters = RetrievalFilters(
        doc_ids=[doc_id] if doc_id else [],
        clause_types=[clause_type] if clause_type and clause_type in BY_SLUG else [],
    )
    chunks, _trace = await retriever.retrieve(query, filters=filters, top_k=top_k)
    if not chunks:
        return json.dumps({"results": [], "note": "No matching excerpts found."})

    return json.dumps(
        {
            "results": [
                {
                    "citation": c.citation_label(),
                    "doc_id": c.doc_id,
                    "section": c.section_path,
                    # Truncated on purpose: tool results re-enter the context on
                    # every subsequent turn of the agent loop, so a fat result
                    # is paid for repeatedly.
                    "text": c.text[:1200],
                    "score": round(c.final_score, 4),
                }
                for c in chunks
            ]
        },
        ensure_ascii=False,
    )


async def list_documents() -> str:
    """List every indexed document with its metadata."""
    from app.ingestion.pipeline import get_pipeline

    docs = get_pipeline().manifest.list()
    return json.dumps(
        {
            "count": len(docs),
            "documents": [
                {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "contract_type": d.get("contract_type"),
                    "governing_law": d.get("governing_law"),
                    "parties": d.get("parties", []),
                    "chunk_count": d.get("chunk_count", 0),
                }
                for d in docs[:100]
            ],
        },
        ensure_ascii=False,
    )


async def get_document_info(doc_id: str) -> str:
    """Metadata for one document: parties, type, governing law, size."""
    from app.ingestion.pipeline import get_pipeline

    doc = get_pipeline().manifest.get(doc_id)
    if doc is None:
        return json.dumps({"error": f"No document with id '{doc_id}'. Call list_documents first."})
    doc.pop("metadata", None)
    return json.dumps(doc, ensure_ascii=False, default=str)


async def list_clause_types() -> str:
    """The 41 clause categories available as metadata filters."""
    return json.dumps({"clause_types": category_labels()}, ensure_ascii=False)


async def compare_clause(clause_type: str, doc_ids: list[str]) -> str:
    """Pull the same clause type from several documents, for comparison."""
    if clause_type not in BY_SLUG:
        return json.dumps(
            {"error": f"Unknown clause_type '{clause_type}'. Call list_clause_types for valid slugs."}
        )
    retriever = get_retriever()
    category = BY_SLUG[clause_type]

    findings: dict[str, Any] = {}
    for doc_id in doc_ids[:6]:  # bound the fan-out
        chunks, _trace = await retriever.retrieve(
            category.description,
            filters=RetrievalFilters(doc_ids=[doc_id], clause_types=[clause_type]),
            top_k=3,
        )
        findings[doc_id] = [
            {"citation": c.citation_label(), "text": c.text[:900]} for c in chunks
        ] or [{"note": "No clause of this type found in this document."}]

    return json.dumps({"clause_type": clause_type, "findings": findings}, ensure_ascii=False)


# ================================================ ANTHROPIC TOOL DEFINITIONS
def _schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic -> JSON Schema, tightened for `strict: True`.

    Strict tool use requires `additionalProperties: false`, which Pydantic does
    not emit by default.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema["additionalProperties"] = False
    schema.setdefault("required", [])
    return schema


ANTHROPIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_contracts",
        "description": (
            "Search the indexed contracts for excerpts relevant to a query. This is the "
            "primary tool - use it for any question about what a contract says. Phrase the "
            "query in contract vocabulary ('termination for convenience' rather than "
            "'cancelling early'). Narrow with clause_type or doc_id when you know them."
        ),
        "strict": True,
        "input_schema": _schema(SearchContractsArgs),
    },
    {
        "name": "list_documents",
        "description": (
            "List every indexed contract with its id, title, type, parties and governing law. "
            "Call this first when the user refers to a document by name, or when you need a "
            "doc_id for another tool."
        ),
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "get_document_info",
        "description": "Get metadata for one document by id: parties, contract type, governing law, size.",
        "strict": True,
        "input_schema": _schema(DocumentArgs),
    },
    {
        "name": "list_clause_types",
        "description": (
            "List the 41 clause categories that can be used as filters. Call this when you are "
            "unsure of the exact slug for a clause type."
        ),
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "compare_clause",
        "description": (
            "Retrieve the same clause type from several documents at once, for comparison. "
            "Use for 'how does X differ between these contracts' questions rather than making "
            "one search_contracts call per document."
        ),
        "strict": True,
        "input_schema": _schema(CompareClauseArgs),
    },
]

TOOL_IMPLEMENTATIONS = {
    "search_contracts": search_contracts,
    "list_documents": list_documents,
    "get_document_info": get_document_info,
    "list_clause_types": list_clause_types,
    "compare_clause": compare_clause,
}


async def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch one tool call, converting failures into readable tool results.

    A raised exception would end the agent loop; an error *string* lets the
    model read what went wrong and try a different call.
    """
    fn = TOOL_IMPLEMENTATIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        return await fn(**arguments)
    except TypeError as exc:
        return json.dumps({"error": f"Bad arguments for {name}: {exc}"})
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return json.dumps({"error": f"{name} failed: {type(exc).__name__}: {exc}"})


# ============================================================ LANGCHAIN TOOLS
def build_langchain_tools() -> list:
    """The same five tools as LangChain `StructuredTool`s (Module 6: Tools)."""
    from langchain_core.tools import StructuredTool

    class _NoArgs(BaseModel):
        pass

    return [
        StructuredTool.from_function(
            coroutine=search_contracts,
            name="search_contracts",
            description=ANTHROPIC_TOOLS[0]["description"],
            args_schema=SearchContractsArgs,
        ),
        StructuredTool.from_function(
            coroutine=list_documents,
            name="list_documents",
            description=ANTHROPIC_TOOLS[1]["description"],
            args_schema=_NoArgs,
        ),
        StructuredTool.from_function(
            coroutine=get_document_info,
            name="get_document_info",
            description=ANTHROPIC_TOOLS[2]["description"],
            args_schema=DocumentArgs,
        ),
        StructuredTool.from_function(
            coroutine=list_clause_types,
            name="list_clause_types",
            description=ANTHROPIC_TOOLS[3]["description"],
            args_schema=_NoArgs,
        ),
        StructuredTool.from_function(
            coroutine=compare_clause,
            name="compare_clause",
            description=ANTHROPIC_TOOLS[4]["description"],
            args_schema=CompareClauseArgs,
        ),
    ]
