"""Golden evaluation set from CUAD (Module 8: Ground Truth Evaluation).

Most RAG projects evaluate with an LLM judging its own output, which measures
plausibility rather than correctness. CUAD gives us something much stronger:
for all 510 contracts, lawyers marked the exact text spans that answer each of
41 clause questions. That turns retrieval evaluation into an objective
measurement - a retrieved chunk either overlaps a lawyer-marked span or it does
not, and no model opinion is involved.

Two things are built here:

  `GoldenExample`   one (question, document, gold spans, reference answer) row.
  span matching     deciding whether a retrieved chunk hit a gold span.

The questions are rewritten rather than reused. CUAD's own prompt is
"Highlight the parts (if any) of this contract related to X" - a span-extraction
instruction aimed at an extractive QA model. A user of this app asks "what is
the governing law?", so the eval asks that too. Evaluating on prompts nobody
would type produces numbers that do not transfer.
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config import Settings, get_settings
from app.ingestion.loaders import make_doc_id, normalise_text
from app.ingestion.taxonomy import BY_CUAD_NAME

logger = logging.getLogger(__name__)

# Natural questions per CUAD category - what a user would actually type.
CATEGORY_QUESTIONS: dict[str, str] = {
    "Document Name": "What type of agreement is this document?",
    "Parties": "Who are the parties to this agreement?",
    "Agreement Date": "What date was this agreement signed?",
    "Effective Date": "When does this agreement become effective?",
    "Expiration Date": "When does the initial term of this agreement expire?",
    "Renewal Term": "How does this agreement renew after the initial term?",
    "Notice Period To Terminate Renewal": "How much notice is required to stop this agreement from renewing?",
    "Governing Law": "Which jurisdiction's law governs this agreement?",
    "Most Favored Nation": "Is there a most favoured nation clause requiring better third-party terms to be passed on?",
    "Non-Compete": "Does this agreement restrict either party from competing?",
    "Exclusivity": "What exclusivity commitments does this agreement impose?",
    "No-Solicit Of Customers": "Are the parties restricted from soliciting each other's customers?",
    "Competitive Restriction Exception": "What exceptions or carve-outs apply to the competitive restrictions?",
    "No-Solicit Of Employees": "Are the parties restricted from hiring or soliciting each other's employees?",
    "Non-Disparagement": "Is there a non-disparagement obligation in this agreement?",
    "Termination For Convenience": "Can either party terminate this agreement without cause?",
    "Rofr/Rofo/Rofn": "Does either party have a right of first refusal, first offer, or first negotiation?",
    "Change Of Control": "What happens under this agreement if one party undergoes a change of control?",
    "Anti-Assignment": "Can this agreement be assigned to a third party, and is consent required?",
    "Revenue/Profit Sharing": "Does either party have to share revenue or profit with the other?",
    "Price Restrictions": "Are there any restrictions on changing prices under this agreement?",
    "Minimum Commitment": "Is there a minimum purchase or volume commitment in this agreement?",
    "Volume Restriction": "Are there fees or consent requirements if usage exceeds a certain volume?",
    "Ip Ownership Assignment": "Who owns intellectual property created under this agreement?",
    "Joint Ip Ownership": "Is any intellectual property jointly owned under this agreement?",
    "License Grant": "What licence does this agreement grant, and to whom?",
    "Non-Transferable License": "Is the licence granted under this agreement transferable or sublicensable?",
    "Affiliate License-Licensor": "Does the licence cover intellectual property held by the licensor's affiliates?",
    "Affiliate License-Licensee": "Does the licence extend to the licensee's affiliates?",
    "Unlimited/All-You-Can-Eat-License": "Does this agreement grant an unlimited or enterprise-wide licence?",
    "Irrevocable Or Perpetual License": "Is the licence granted under this agreement irrevocable or perpetual?",
    "Source Code Escrow": "Does this agreement require source code to be placed in escrow?",
    "Post-Termination Services": "What obligations survive termination or expiry of this agreement?",
    "Audit Rights": "Does either party have the right to audit the other's books or records?",
    "Uncapped Liability": "Is either party's liability uncapped in any circumstances?",
    "Cap On Liability": "Is there a cap on liability, and what is it?",
    "Liquidated Damages": "Does this agreement provide for liquidated damages on breach?",
    "Warranty Duration": "How long do the warranties in this agreement last?",
    "Insurance": "What insurance is either party required to maintain?",
    "Covenant Not To Sue": "Is either party restricted from challenging the other's intellectual property?",
    "Third Party Beneficiary": "Are there any third-party beneficiaries of this agreement?",
}

# Below this length a span is too generic to match reliably ("Distributor"),
# so it is matched as an exact substring rather than by fuzzy overlap.
SHORT_SPAN_CHARS = 60
SHINGLE_SIZE = 8
OVERLAP_THRESHOLD = 0.35


@dataclass(slots=True)
class GoldenExample:
    example_id: str
    question: str
    category: str              # exact CUAD column name
    category_slug: str | None
    doc_id: str
    doc_title: str
    gold_spans: list[str] = field(default_factory=list)
    reference_answer: str = ""
    is_negative: bool = False  # the contract genuinely has no such clause

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "category": self.category,
            "category_slug": self.category_slug,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "gold_span_count": len(self.gold_spans),
            "reference_answer": self.reference_answer,
            "is_negative": self.is_negative,
        }


# --------------------------------------------------------------- matching
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = _norm(text).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def span_matches_chunk(span: str, chunk_text: str) -> bool:
    """Did this chunk capture the lawyer-marked span?

    Overlap has to work in both directions: a gold span can be longer than one
    chunk (a three-paragraph indemnity split across two chunks) or much shorter
    (a signature date inside a large chunk). Word-shingle overlap scored against
    the *shorter* side handles both; a strict containment test would score a
    correct retrieval as a miss in the first case.
    """
    span_norm, chunk_norm = _norm(span), _norm(chunk_text)
    if not span_norm or not chunk_norm:
        return False

    if len(span_norm) < SHORT_SPAN_CHARS:
        return span_norm in chunk_norm

    if span_norm in chunk_norm or chunk_norm in span_norm:
        return True

    span_shingles, chunk_shingles = _shingles(span), _shingles(chunk_text)
    if not span_shingles or not chunk_shingles:
        return False
    shared = len(span_shingles & chunk_shingles)
    return shared / min(len(span_shingles), len(chunk_shingles)) >= OVERLAP_THRESHOLD


def chunk_is_relevant(chunk: dict[str, Any], example: GoldenExample) -> bool:
    """A chunk is relevant when it is from the right document AND overlaps a span."""
    if chunk.get("doc_id") != example.doc_id:
        return False
    text = chunk.get("text", "")
    return any(span_matches_chunk(span, text) for span in example.gold_spans)


# ---------------------------------------------------------------- loading
class GoldenSet:
    def __init__(self, examples: list[GoldenExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def sample(
        self,
        n: int,
        *,
        seed: int = 42,
        categories: Iterable[str] | None = None,
        doc_ids: Iterable[str] | None = None,
        include_negatives: bool = False,
    ) -> list[GoldenExample]:
        """Stratified sample - spread across categories, not clustered in one.

        A random sample of 25 from 20,000 rows can easily land mostly on
        Governing Law and tell you nothing about the other 40 categories.
        """
        pool = self.examples
        if categories:
            wanted = set(categories)
            pool = [e for e in pool if e.category in wanted or e.category_slug in wanted]
        if doc_ids:
            allowed = set(doc_ids)
            pool = [e for e in pool if e.doc_id in allowed]
        if not include_negatives:
            pool = [e for e in pool if not e.is_negative]
        if not pool:
            return []

        by_category: dict[str, list[GoldenExample]] = {}
        for example in pool:
            by_category.setdefault(example.category, []).append(example)

        rng = random.Random(seed)
        for bucket in by_category.values():
            rng.shuffle(bucket)

        ordered_categories = sorted(by_category)
        selected: list[GoldenExample] = []
        round_index = 0
        while len(selected) < n:
            added = False
            for category in ordered_categories:
                bucket = by_category[category]
                if round_index < len(bucket):
                    selected.append(bucket[round_index])
                    added = True
                    if len(selected) >= n:
                        break
            if not added:
                break
            round_index += 1
        return selected[:n]

    def by_document(self) -> dict[str, list[GoldenExample]]:
        out: dict[str, list[GoldenExample]] = {}
        for example in self.examples:
            out.setdefault(example.doc_id, []).append(example)
        return out


def build_golden_set(
    settings: Settings | None = None,
    *,
    doc_limit: int | None = None,
    include_negatives: bool = True,
) -> GoldenSet:
    """Build the golden set from CUAD_v1.json + master_clauses.csv.

    `doc_id` is recomputed with the same content hash the ingestion pipeline
    uses, so examples join to indexed chunks without a lookup table.
    """
    settings = settings or get_settings()
    json_path = settings.cuad_dir / "CUAD_v1.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"CUAD_v1.json not found at {json_path}. Set CUAD_DIR in .env."
        )

    reference_answers = _load_reference_answers(settings)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    entries = payload["data"][:doc_limit] if doc_limit else payload["data"]

    examples: list[GoldenExample] = []
    for entry in entries:
        title = entry["title"]
        context = entry["paragraphs"][0]["context"]
        # Hash the normalised text - exactly what the ingestion pipeline hashes.
        doc_id = make_doc_id(f"{title}.txt", normalise_text(context))
        doc_answers = reference_answers.get(title, {})

        for qa in entry["paragraphs"][0]["qas"]:
            category = qa["id"].split("__")[-1]
            question = CATEGORY_QUESTIONS.get(category)
            if question is None:
                continue
            spans = [a["text"] for a in qa.get("answers", []) if a.get("text", "").strip()]
            is_negative = not spans
            if is_negative and not include_negatives:
                continue
            taxonomy_entry = BY_CUAD_NAME.get(category)
            examples.append(
                GoldenExample(
                    example_id=qa["id"],
                    question=question,
                    category=category,
                    category_slug=taxonomy_entry.slug if taxonomy_entry else None,
                    doc_id=doc_id,
                    doc_title=title,
                    gold_spans=spans,
                    reference_answer=str(doc_answers.get(category, "")).strip(),
                    is_negative=is_negative,
                )
            )

    positives = sum(1 for e in examples if not e.is_negative)
    logger.info(
        "Golden set: %d examples over %d documents (%d positive, %d negative)",
        len(examples), len(entries), positives, len(examples) - positives,
    )
    return GoldenSet(examples)


def _load_reference_answers(settings: Settings) -> dict[str, dict[str, str]]:
    """The human-written short answers from master_clauses.csv ('Germany', 'Yes')."""
    csv_path = settings.cuad_dir / "master_clauses.csv"
    if not csv_path.exists():
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}

    frame = pd.read_csv(csv_path)
    out: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        stem = Path(str(row["Filename"])).stem
        answers: dict[str, str] = {}
        for column in frame.columns:
            if not column.endswith("-Answer") and not column.endswith("- Answer"):
                continue
            category = column.replace("- Answer", "").replace("-Answer", "").strip()
            value = row[column]
            if isinstance(value, str) and value.strip():
                answers[category] = value.strip()
        out[stem] = answers
    return out


_cached: GoldenSet | None = None


def get_golden_set(settings: Settings | None = None, **kwargs) -> GoldenSet:
    global _cached
    if _cached is None:
        _cached = build_golden_set(settings, **kwargs)
    return _cached
