"""Guardrails (Tier 1 M3: Agent Guardrails + Module 8: Response Validation).

Guardrails run on both sides of the model, and the two sides have genuinely
different jobs.

Input guardrails are cheap, deterministic and run first: length limits, prompt
injection detection, and a fast keyword screen for advice-seeking. They exist to
stop a request before it costs money.

Output guardrails validate what came back. The important one here is **citation
validation**: the answer prompt promises that every claim carries a [n] pointing
at a real excerpt, and this checks that promise mechanically. A model that cites
[7] when it was given 6 excerpts has hallucinated its own evidence, and no
amount of LLM judging is needed to catch that - it is a parsing problem.

Deliberate design choice: guardrails *annotate* rather than silently rewrite.
A blocked answer is replaced, but a low-confidence one is passed through with a
warning attached, because hiding a weak answer is worse for a research tool
than showing it with its weakness visible.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(\d{1,2})\]")

# Attempts to override the system prompt. Not exhaustive - nothing is - but it
# catches the common shapes, and the architecture behind it matters more: the
# user's text only ever enters the *user* turn, never the system prompt, so an
# injected instruction is competing with a cached system prompt rather than
# becoming part of it.
INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions?\b", re.I),
    re.compile(r"\bdisregard\s+(?:your|the|all)\s+(?:instructions?|rules?|guidelines?|system prompt)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(?:a|an)\b.{0,40}\b(?:lawyer|attorney|unrestricted|DAN)\b", re.I),
    re.compile(r"\b(?:reveal|show|print|repeat)\s+(?:your|the)\s+(?:system\s+prompt|instructions?)\b", re.I),
    re.compile(r"\bpretend\s+(?:you\s+are|to\s+be)\b.{0,30}\b(?:not bound|unrestricted|no rules)\b", re.I),
    re.compile(r"</?(?:system|instructions?)>", re.I),
)

# Advice-seeking, as opposed to fact-seeking. The distinction that matters:
# "what does the termination clause say" is research; "should I sign this" is
# advice. The LLM router makes the final call - this is the cheap pre-screen.
ADVICE_PATTERNS = (
    re.compile(r"\bshould\s+I\b", re.I),
    re.compile(r"\b(?:can|could|would)\s+I\s+(?:sue|win|be liable|get sued)\b", re.I),
    # Allow an intervening noun: "is this *clause* enforceable" is the same
    # question as "is this enforceable".
    re.compile(
        r"\bis\s+(?:this|it|that|the)\s+(?:\w+\s+){0,2}"
        r"(?:legal|illegal|enforceable|unenforceable|binding|valid|invalid|fair|"
        r"a\s+good\s+deal|standard|normal|reasonable)\b",
        re.I,
    ),
    re.compile(r"\bwhat\s+(?:should|would)\s+(?:I|we|you)\s+do\b", re.I),
    re.compile(r"\bdo\s+I\s+(?:have\s+to|need\s+to)\s+(?:sign|accept|agree)\b", re.I),
    re.compile(r"\bhow\s+(?:would|will)\s+a?\s*(?:court|judge)\b", re.I),
    re.compile(r"\badvise\s+me\b", re.I),
    re.compile(r"\bwhat\s+are\s+my\s+(?:legal\s+)?(?:rights|options|chances)\b", re.I),
)


@dataclass(slots=True)
class GuardrailVerdict:
    allowed: bool = True
    reason: str | None = None
    category: str | None = None
    warnings: list[str] = field(default_factory=list)
    replacement: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "category": self.category,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


# ================================================================ INPUT SIDE
def check_input(question: str, settings: Settings | None = None) -> GuardrailVerdict:
    settings = settings or get_settings()
    verdict = GuardrailVerdict()
    if not settings.guardrails_enabled:
        return verdict

    text = (question or "").strip()

    if not text:
        return GuardrailVerdict(
            allowed=False, reason="Empty question.", category="empty",
            replacement="Ask me something about the indexed contracts and I'll find the relevant clauses.",
        )

    if len(text) > settings.max_question_chars:
        return GuardrailVerdict(
            allowed=False,
            reason=f"Question exceeds {settings.max_question_chars} characters.",
            category="too_long",
            replacement=(
                f"That question is {len(text):,} characters, past the "
                f"{settings.max_question_chars:,} limit. Could you narrow it to the "
                "specific clause or issue you need? If you meant to add a document, "
                "use the upload button instead of pasting it."
            ),
        )

    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("Prompt injection pattern matched: %.80s", text)
            return GuardrailVerdict(
                allowed=False, reason="Possible prompt injection.", category="injection",
                replacement=(
                    "I can only answer questions about the contracts that are indexed here. "
                    "Ask me what a document says about a particular issue and I'll pull the "
                    "relevant clauses with citations."
                ),
            )

    # Advice is flagged, not blocked - the LLM router decides, because these
    # patterns over-trigger ("should I look at section 5?" is not advice).
    if any(p.search(text) for p in ADVICE_PATTERNS):
        verdict.warnings.append("possible_legal_advice")
        verdict.metadata["advice_prescreen"] = True

    return verdict


# =============================================================== OUTPUT SIDE
def validate_citations(
    answer: str, chunks: Sequence[RetrievedChunk]
) -> tuple[list[int], list[int], float]:
    """Return (valid, invalid, coverage) citation indices found in the answer.

    `coverage` is the share of supplied excerpts the answer actually used -
    useful as a retrieval signal: if the answer only ever cites [1], the other
    five chunks were wasted context.
    """
    cited = [int(m.group(1)) for m in CITATION_RE.finditer(answer)]
    valid = sorted({n for n in cited if 1 <= n <= len(chunks)})
    invalid = sorted({n for n in cited if n < 1 or n > len(chunks)})
    coverage = len(valid) / len(chunks) if chunks else 0.0
    return valid, invalid, coverage


def check_output(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    *,
    settings: Settings | None = None,
    expect_citations: bool = True,
) -> GuardrailVerdict:
    settings = settings or get_settings()
    verdict = GuardrailVerdict()
    if not settings.guardrails_enabled:
        return verdict

    valid, invalid, coverage = validate_citations(answer, chunks)
    verdict.metadata.update(
        {
            "citations_valid": valid,
            "citations_invalid": invalid,
            "citation_coverage": round(coverage, 3),
        }
    )

    # A citation pointing at an excerpt that does not exist is fabricated
    # evidence. This is the one output failure we treat as hard.
    if invalid:
        verdict.warnings.append("invalid_citations")
        verdict.allowed = False
        verdict.category = "invalid_citation"
        verdict.reason = f"Answer cited non-existent excerpts: {invalid}"
        return verdict

    # An answer that says the excerpts lack the information is correct
    # behaviour and legitimately carries no citations.
    disclaims = re.search(
        r"\b(?:do(?:es)? not (?:address|contain|state|specify|cover)|not (?:covered|included|present|shown)|no (?:excerpt|passage|clause)|could not find|unable to find)\b",
        answer,
        re.I,
    )
    if expect_citations and chunks and not valid and not disclaims:
        verdict.warnings.append("no_citations")
        verdict.metadata["citation_warning"] = (
            "This answer makes claims without pointing to a specific clause."
        )

    if re.search(r"\b(?:I (?:would )?(?:recommend|advise|suggest you)|you should (?:sign|accept|refuse|sue))\b", answer, re.I):
        verdict.warnings.append("advisory_language")

    return verdict


def build_refusal(category: str) -> str:
    from app.prompts.templates import LEGAL_ADVICE_REFUSAL, NO_RESULTS_RESPONSE

    return {
        "legal_advice": LEGAL_ADVICE_REFUSAL,
        "no_results": NO_RESULTS_RESPONSE,
        "out_of_scope": (
            "I only work with the contracts indexed in this workspace, so I can't help "
            "with that. Ask me about a clause, an obligation, a date, or a restriction "
            "in one of the documents and I'll pull it with a citation."
        ),
    }.get(category, NO_RESULTS_RESPONSE)
