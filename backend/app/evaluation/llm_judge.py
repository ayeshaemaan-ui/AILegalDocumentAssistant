"""LLM-as-judge (Module 8: LLM Evaluation Metrics + Response Validation).

Retrieval can be scored objectively against CUAD's spans. Answer *quality*
cannot - there is no string comparison that tells you whether "Either party may
terminate on 30 days' notice" is a good answer to "can I exit early?". So a
model grades it against the lawyer-written reference.

Three deliberate choices keep this honest:

  A cheaper, different-role model grades. Haiku judging Opus is not the same
  model marking its own homework, and it costs a fraction of a cent per grade.

  The judge sees the reference answer. Judging without a reference measures
  how confident the answer sounds; judging against a lawyer's answer measures
  whether it is right.

  Scores are structured, not prose. `AnswerGrade` forces integers 1-5 on three
  separate axes, so "correct but missing the exception" is representable
  instead of collapsing into one vague verdict.

The scores are still a proxy. They are useful for comparing two versions of
this system against each other - which is what evaluation is for - and not for
claiming an absolute quality level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.core.llm import Usage, get_llm_client
from app.graph.schemas import AnswerGrade, GroundednessReport
from app.prompts import templates
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JudgeResult:
    correctness: int = 0
    completeness: int = 0
    citation_quality: int = 0
    reasoning: str = ""
    failed: bool = False

    @property
    def overall(self) -> float:
        if self.failed:
            return 0.0
        return (self.correctness + self.completeness + self.citation_quality) / 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "citation_quality": self.citation_quality,
            "overall": round(self.overall, 3),
            "reasoning": self.reasoning,
            "failed": self.failed,
        }


class AnswerJudge:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def grade(
        self,
        *,
        question: str,
        answer: str,
        reference: str,
        gold_spans: Sequence[str] | None = None,
    ) -> tuple[JudgeResult, Usage]:
        client = get_llm_client(self.settings)

        # The lawyer-marked spans are the evidence behind the reference answer,
        # so showing them lets the judge distinguish "wrong" from "phrased
        # differently but supported by the same clause".
        evidence = ""
        if gold_spans:
            joined = "\n".join(f"- {span[:400]}" for span in list(gold_spans)[:4])
            evidence = f"\n\n## Clauses the reference was drawn from\n{joined}"

        user_turn = (
            f"## Question\n{question}\n\n"
            f"## Reference answer (written by a lawyer)\n{reference or '(none provided)'}"
            f"{evidence}\n\n"
            f"## Answer to grade\n{answer}"
        )
        try:
            grade, usage = await client.structured(
                system=templates.ANSWER_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_turn}],
                schema=AnswerGrade,
                model=self.settings.resolved_judge_model,
                effort="low",
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning("Judge failed: %s", exc)
            return JudgeResult(reasoning=f"judge unavailable: {exc}", failed=True), Usage()

        return (
            JudgeResult(
                correctness=grade.correctness,
                completeness=grade.completeness,
                citation_quality=grade.citation_quality,
                reasoning=grade.reasoning,
            ),
            usage,
        )


class FaithfulnessJudge:
    """Groundedness scoring, reused outside the graph for batch evaluation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def score(
        self, *, question: str, answer: str, chunks: Sequence[RetrievedChunk]
    ) -> tuple[dict[str, Any], Usage]:
        if not answer.strip():
            return {"verdict": "grounded", "grounded_ratio": 1.0, "reasoning": "Empty answer."}, Usage()

        client = get_llm_client(self.settings)
        try:
            report, usage = await client.structured(
                system=templates.GROUNDEDNESS_SYSTEM,
                messages=[
                    {"role": "user", "content": templates.build_groundedness_turn(question, answer, chunks)}
                ],
                schema=GroundednessReport,
                model=self.settings.resolved_judge_model,
                effort="low",
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning("Faithfulness check failed: %s", exc)
            return {"verdict": "unknown", "grounded_ratio": -1.0, "reasoning": str(exc)}, Usage()
        return report.model_dump(), usage
