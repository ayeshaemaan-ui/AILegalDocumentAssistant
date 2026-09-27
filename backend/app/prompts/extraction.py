"""Prompts for clause extraction.

Deliberately narrower than the chat prompts. Extraction is not a conversation:
it has one job, one output shape, and the most common failure is a model that
would rather invent a plausible value than admit the clause is absent. Every
instruction below exists to make abstaining the easy path.
"""
from __future__ import annotations

from typing import Sequence

from app.retrieval.types import RetrievedChunk

EXTRACTION_SYSTEM = """\
You extract a single named clause value from excerpts of one commercial contract.

You are building a row in a review table that a person will check. Your output \
is only useful if it is either correct or honestly marked as uncertain.

Rules, in priority order:

1. Use ONLY the excerpts provided. Never use general knowledge about what \
contracts usually say. If the excerpts do not contain it, you do not know it.

2. Choose the status honestly:
   - "found" only when an excerpt states the answer.
   - "not_present" when the excerpts clearly show this contract has no such \
clause (for example, a full liability section that contains no cap).
   - "unsure" when the excerpts are ambiguous, partial, or about a different \
clause. Choosing "unsure" is always better than guessing.

3. Keep `value` short and literal. "Delaware", not "The laws of the State of \
Delaware shall govern". "90 days", not "ninety (90) days written notice". \
For yes/no clauses, answer "Yes" or "No".

4. `quote` must be copied character-for-character from an excerpt. Do not \
paraphrase, tidy, or shorten it. A reviewer will search for this text.

5. Set `confidence` to "low" whenever you had to interpret rather than read.\
"""


def build_extraction_turn(
    *,
    clause_label: str,
    clause_description: str,
    document_title: str,
    chunks: Sequence[RetrievedChunk],
) -> str:
    if not chunks:
        return (
            f"## Clause to extract\n{clause_label}\n\n"
            f"## Contract\n{document_title}\n\n"
            "## Excerpts\n(none were retrieved)\n\n"
            "No excerpts were found, so the status must be 'unsure'."
        )

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        location = chunk.metadata.get("section_path") or chunk.metadata.get("section_number") or ""
        header = f"### Excerpt [{i}]" + (f" — {location}" if location else "")
        blocks.append(f"{header}\n{chunk.text.strip()}")

    return (
        f"## Clause to extract\n{clause_label}\n{clause_description}\n\n"
        f"## Contract\n{document_title}\n\n"
        f"## Excerpts\n\n" + "\n\n".join(blocks) + "\n\n"
        f"Extract the {clause_label} for this contract, following the rules."
    )
