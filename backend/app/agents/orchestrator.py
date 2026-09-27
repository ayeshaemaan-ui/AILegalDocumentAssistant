"""Agents and orchestration (Module 9: Agentic AI + Tier 1 M3: Multi-Step Agents).

Two levels, and the distinction between them is the whole point of the module.

`ResearchAgent` is a single agent: a tool-calling loop where the model chooses
its next search based on what the last one returned. Use it when a question
needs several dependent lookups - "which of these contracts has the shortest
termination notice" requires listing documents, then searching each, then
comparing.

`Orchestrator` is the layer above: it decomposes a broad question into
independent sub-questions, runs a sub-agent per sub-question **concurrently**,
then synthesises one answer from their findings. The reason to do this rather
than let one agent grind through sequentially is context, not speed - each
sub-agent reads only its own search results, so none of them drowns in a
context window full of other sub-questions' excerpts.

Guardrails are structural, not advisory (Tier 1 M3: Agent Guardrails):
  * `max_steps` bounds the loop - an agent that cannot finish must stop.
  * `max_cost_usd` bounds spend per run, checked after every model call.
  * every tool is read-only, so a confused loop cannot damage anything.
  * `fan_out` is capped, so decomposition cannot spawn unbounded sub-agents.
  * the final answer still goes through the same output guardrails as the graph.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from app.agents.tools import ANTHROPIC_TOOLS, execute_tool
from app.config import Settings, get_settings
from app.core.cost import get_cost_tracker
from app.core.llm import Usage, get_llm_client

logger = logging.getLogger(__name__)

AGENT_SYSTEM = """You are a contract research agent. You answer questions by searching \
indexed contracts with the tools you have been given.

How to work:
1. Plan before searching. If the question spans several documents or several clause \
types, work out what you need to look up before you start.
2. Search with the contract's vocabulary, not the user's. "Cancelling early" is \
"termination for convenience"; "who can sue" is "third party beneficiary".
3. When a search returns nothing useful, change the wording rather than repeating it. \
If two rewordings both fail, accept that the contracts may be silent and say so.
4. Use `compare_clause` for cross-document questions instead of one search per document.
5. Stop searching once you can answer. Extra searches cost money and add nothing.

How to answer:
- Ground every factual claim in something a tool returned, and cite the clause using \
the citation string from the tool result, like (Distributor Agreement · § 9.2).
- If the tools did not surface something, say so. Never fill the gap from general \
legal knowledge.
- You are a research tool, not a lawyer. Describe what the contracts say; do not \
advise on what to do, whether to sign, or how a dispute would resolve."""

DECOMPOSE_SYSTEM = """You break a broad contract-research question into independent \
sub-questions that can each be researched separately and in parallel.

Rules:
- Each sub-question must be self-contained and answerable on its own. A sub-question \
that depends on another's answer defeats the purpose.
- Produce between 1 and 4. If the question is already a single lookup, return exactly \
one sub-question: the original.
- Cover the original question completely between them - no gaps, no overlap.

Example
  "How do the distributor agreements handle termination and exclusivity?"
  -> "What termination rights do the distributor agreements grant each party?"
  -> "What exclusivity commitments do the distributor agreements impose?" """

SYNTHESIS_SYSTEM = """You write one answer from the findings of several research \
sub-agents.

Merge their findings into a single coherent answer. Keep every citation exactly as the \
sub-agent wrote it. Where sub-agents disagree or one found nothing, say so plainly \
rather than smoothing it over. Do not add facts that no sub-agent reported.

Lead with the direct answer. You are describing what the contracts say - not advising \
on what to do about it."""


class Decomposition(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=4)
    reasoning: str = Field(max_length=300)


@dataclass(slots=True)
class AgentStep:
    step: int
    kind: str                      # "tool_call" | "answer" | "guardrail"
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    elapsed_ms: float = 0.0


@dataclass(slots=True)
class AgentRun:
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    stopped_reason: str = "completed"
    sub_runs: list["AgentRun"] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "stopped_reason": self.stopped_reason,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "steps": [
                {
                    "step": s.step, "kind": s.kind, "name": s.name,
                    "arguments": s.arguments, "result_preview": s.result_preview,
                    "elapsed_ms": round(s.elapsed_ms, 1),
                }
                for s in self.steps
            ],
            "sub_runs": [r.as_dict() for r in self.sub_runs],
        }


class ResearchAgent:
    """A bounded tool-calling loop over the contract corpus."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        max_steps: int = 8,
        max_cost_usd: float = 0.50,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_steps = max_steps
        self.max_cost_usd = max_cost_usd

    async def run(
        self, question: str, *, session_id: str | None = None, trace_id: str | None = None
    ) -> AgentRun:
        client = get_llm_client(self.settings)
        tracker = get_cost_tracker(self.settings)
        run = AgentRun(question=question, answer="")
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        for step in range(1, self.max_steps + 1):
            # -- guardrail: spend ceiling ---------------------------------
            if run.total_cost_usd >= self.max_cost_usd:
                run.stopped_reason = "cost_limit"
                run.steps.append(
                    AgentStep(step, "guardrail", "cost_limit",
                              result_preview=f"${run.total_cost_usd:.4f} of ${self.max_cost_usd:.2f}")
                )
                break

            started = time.perf_counter()
            try:
                result = await client.complete(
                    system=AGENT_SYSTEM,
                    messages=messages,
                    model=self.settings.resolved_answer_model,
                    max_tokens=self.settings.answer_max_tokens,
                    effort=self.settings.answer_effort,
                    tools=ANTHROPIC_TOOLS,
                )
            except Exception as exc:
                logger.exception("Agent step %d failed", step)
                run.answer = f"The research agent hit an error: {type(exc).__name__}: {exc}"
                run.stopped_reason = "error"
                break

            run.total_cost_usd += tracker.record(
                result.usage, stage="agent.step", session_id=session_id, trace_id=trace_id
            )
            run.total_tokens += result.usage.total_tokens

            # -- model is done --------------------------------------------
            if result.stop_reason != "tool_use":
                run.answer = result.text
                run.steps.append(
                    AgentStep(step, "answer", result_preview=result.text[:200],
                              elapsed_ms=(time.perf_counter() - started) * 1000)
                )
                break

            # -- execute the requested tools ------------------------------
            messages.append({"role": "assistant", "content": result.raw.content})

            # Parallel tool calls in one assistant turn must come back as
            # tool_result blocks in ONE user message, or the model learns to
            # stop batching them.
            tool_results: list[dict[str, Any]] = []
            outputs = await asyncio.gather(
                *(execute_tool(call["name"], call["input"]) for call in result.tool_calls)
            )
            for call, output in zip(result.tool_calls, outputs):
                run.steps.append(
                    AgentStep(
                        step, "tool_call", call["name"], call["input"], output[:200],
                        (time.perf_counter() - started) * 1000,
                    )
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": call["id"], "content": output}
                )
            messages.append({"role": "user", "content": tool_results})
        else:
            run.stopped_reason = "max_steps"

        if not run.answer:
            run.answer = (
                "I could not complete the research within the step budget. "
                "A narrower question - one clause or one document at a time - usually works."
            )
        return run


class Orchestrator:
    """Decompose -> fan out to sub-agents -> synthesise (Module 9: Orchestration Layer)."""

    def __init__(
        self, settings: Settings | None = None, *, fan_out: int = 4, max_cost_usd: float = 2.00
    ) -> None:
        self.settings = settings or get_settings()
        self.fan_out = fan_out
        self.max_cost_usd = max_cost_usd

    async def decompose(self, question: str) -> Decomposition:
        client = get_llm_client(self.settings)
        try:
            decomposition, usage = await client.structured(
                system=DECOMPOSE_SYSTEM,
                messages=[{"role": "user", "content": question}],
                schema=Decomposition,
                model=self.settings.resolved_fast_model,
                effort="low",
            )
            get_cost_tracker(self.settings).record(usage, stage="orchestrator.decompose")
            return decomposition
        except Exception as exc:
            logger.warning("Decomposition failed (%s); treating as a single question", exc)
            return Decomposition(sub_questions=[question], reasoning="decomposition unavailable")

    async def run(
        self, question: str, *, session_id: str | None = None, trace_id: str | None = None
    ) -> AgentRun:
        decomposition = await self.decompose(question)
        sub_questions = decomposition.sub_questions[: self.fan_out]

        # One sub-question means no orchestration is warranted - run the agent
        # directly rather than paying for a synthesis call that adds nothing.
        if len(sub_questions) == 1:
            run = await ResearchAgent(self.settings).run(
                sub_questions[0], session_id=session_id, trace_id=trace_id
            )
            run.question = question
            return run

        budget_each = self.max_cost_usd / max(len(sub_questions), 1)
        sub_runs = await asyncio.gather(
            *(
                ResearchAgent(self.settings, max_steps=6, max_cost_usd=budget_each).run(
                    sub_question, session_id=session_id, trace_id=trace_id
                )
                for sub_question in sub_questions
            )
        )

        findings = "\n\n".join(
            f"### Sub-question {i}: {r.question}\n{r.answer}" for i, r in enumerate(sub_runs, 1)
        )
        client = get_llm_client(self.settings)
        try:
            result = await client.complete(
                system=SYNTHESIS_SYSTEM,
                messages=[
                    {"role": "user", "content": f"Original question: {question}\n\n{findings}"}
                ],
                model=self.settings.resolved_answer_model,
                max_tokens=self.settings.answer_max_tokens,
                effort=self.settings.answer_effort,
            )
            answer = result.text
            synthesis_cost = get_cost_tracker(self.settings).record(
                result.usage, stage="orchestrator.synthesize", session_id=session_id, trace_id=trace_id
            )
            synthesis_tokens = result.usage.total_tokens
        except Exception as exc:
            logger.exception("Synthesis failed")
            answer = f"Sub-agent findings (synthesis failed: {exc}):\n\n{findings}"
            synthesis_cost, synthesis_tokens = 0.0, 0

        return AgentRun(
            question=question,
            answer=answer,
            steps=[
                AgentStep(0, "decompose", "orchestrator",
                          {"sub_questions": sub_questions}, decomposition.reasoning[:200])
            ],
            total_cost_usd=sum(r.total_cost_usd for r in sub_runs) + synthesis_cost,
            total_tokens=sum(r.total_tokens for r in sub_runs) + synthesis_tokens,
            sub_runs=list(sub_runs),
        )
