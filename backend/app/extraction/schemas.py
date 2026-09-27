"""Schemas for clause extraction (Module 2: Structured Outputs).

The matrix is only useful if every cell is trustworthy, which means the model
has to be able to say three different things:

  found        here is the value, and here is the sentence it came from
  not present  the contract genuinely does not have this clause
  unsure       there is something here but it does not clearly answer

Collapsing "not present" and "unsure" into an empty cell would be the easy
choice and the wrong one - a reviewer needs to know the difference between "this
contract has no liability cap" and "I could not tell". So `status` is a required
enum rather than an optional note.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClauseExtraction(BaseModel):
    """One cell of the matrix: one clause type, in one contract."""

    status: Literal["found", "not_present", "unsure"] = Field(
        description=(
            "'found' when the excerpts state the clause. 'not_present' when they "
            "clearly show the contract has no such clause. 'unsure' when the "
            "excerpts are ambiguous or off-topic - never guess."
        )
    )
    value: str = Field(
        default="",
        description=(
            "The answer in as few words as possible - 'Delaware', '$2,000,000', "
            "'90 days', 'Yes'. Empty unless status is 'found'."
        ),
    )
    quote: str = Field(
        default="",
        description=(
            "The exact sentence from the excerpts that supports the value, copied "
            "verbatim so a reviewer can verify it. Empty unless status is 'found'."
        ),
    )
    excerpt_index: int = Field(
        default=0,
        description="Which numbered excerpt the quote came from. 0 if none.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="How clearly the excerpts support this. 'low' invites review.",
    )
