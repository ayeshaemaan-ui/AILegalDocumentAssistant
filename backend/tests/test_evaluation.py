"""Golden-set construction and span matching."""
from __future__ import annotations

import pytest

from app.evaluation.golden_set import (
    CATEGORY_QUESTIONS,
    GoldenExample,
    GoldenSet,
    chunk_is_relevant,
    span_matches_chunk,
)


class TestSpanMatching:
    def test_exact_containment(self):
        span = "Supplier may terminate this Agreement at any time upon ninety days notice."
        chunk = f"3.2 Termination for Convenience. {span} Distributor shall be notified."
        assert span_matches_chunk(span, chunk)

    def test_whitespace_differences_do_not_matter(self):
        assert span_matches_chunk("governed by Delaware law", "This   Agreement is\n governed by  Delaware law.")

    def test_span_longer_than_chunk_still_matches(self):
        """A long clause split across chunks must still count as retrieved -
        strict containment would score a correct retrieval as a miss."""
        span = (
            "In no event shall either party be liable for indirect, incidental, special, "
            "consequential or punitive damages arising out of or relating to this Agreement, "
            "whether in contract, tort or otherwise, even if advised of the possibility of such damages."
        )
        chunk = (
            "In no event shall either party be liable for indirect, incidental, special, "
            "consequential or punitive damages arising out of or relating to this Agreement,"
        )
        assert span_matches_chunk(span, chunk)

    def test_unrelated_text_does_not_match(self):
        span = "The Agreement is governed by the laws of the State of Delaware."
        chunk = "Distributor shall purchase at least five hundred units each calendar year."
        assert not span_matches_chunk(span, chunk)

    def test_short_spans_require_exact_substring(self):
        """Short spans like a date are too generic for fuzzy overlap."""
        assert span_matches_chunk("March 14, 2021", "dated as of March 14, 2021 between")
        assert not span_matches_chunk("March 14, 2021", "dated as of April 2, 2019 between")

    def test_empty_inputs(self):
        assert not span_matches_chunk("", "text")
        assert not span_matches_chunk("text", "")


class TestChunkRelevance:
    def test_requires_the_right_document(self):
        """A matching clause in the wrong contract is not a correct retrieval."""
        example = GoldenExample(
            example_id="e1", question="q", category="Governing Law", category_slug="governing_law",
            doc_id="doc1", doc_title="t",
            gold_spans=["This Agreement shall be governed by the laws of the State of Delaware."],
        )
        text = "This Agreement shall be governed by the laws of the State of Delaware."
        assert chunk_is_relevant({"doc_id": "doc1", "text": text}, example)
        assert not chunk_is_relevant({"doc_id": "doc2", "text": text}, example)


class TestGoldenSet:
    @pytest.fixture
    def golden(self):
        examples = []
        for category in ["Governing Law", "Insurance", "Audit Rights", "Cap On Liability"]:
            for doc in range(5):
                examples.append(
                    GoldenExample(
                        example_id=f"{category}-{doc}",
                        question=CATEGORY_QUESTIONS[category],
                        category=category,
                        category_slug=None,
                        doc_id=f"doc{doc}",
                        doc_title=f"Contract {doc}",
                        gold_spans=["some clause text long enough to be meaningful"],
                    )
                )
        examples.append(
            GoldenExample("neg", "q", "Insurance", None, "doc0", "Contract 0", [], is_negative=True)
        )
        return GoldenSet(examples)

    def test_stratification(self, golden):
        """A plain random sample can land entirely in one category and tell you
        nothing about the other forty, so sampling walks the categories."""
        sample = golden.sample(8, seed=1)
        assert len(sample) == 8
        assert len({e.category for e in sample}) == 4

    def test_sampling_is_deterministic(self, golden):
        a = [e.example_id for e in golden.sample(6, seed=7)]
        b = [e.example_id for e in golden.sample(6, seed=7)]
        assert a == b

    def test_negatives_excluded_by_default(self, golden):
        assert all(not e.is_negative for e in golden.sample(30, seed=1))

    def test_category_filter(self, golden):
        sample = golden.sample(10, seed=1, categories=["Insurance"])
        assert {e.category for e in sample} == {"Insurance"}

    def test_document_filter_scopes_to_indexed_docs(self, golden):
        """Scoring questions about documents that were never indexed would
        report 0.0 recall for a retrieval failure that did not happen."""
        sample = golden.sample(20, seed=1, doc_ids={"doc0", "doc1"})
        assert {e.doc_id for e in sample} <= {"doc0", "doc1"}

    def test_asking_for_more_than_exists(self, golden):
        assert len(golden.sample(500, seed=1)) == 20

    def test_every_category_has_a_natural_question(self):
        assert len(CATEGORY_QUESTIONS) == 41
        for question in CATEGORY_QUESTIONS.values():
            # CUAD's own prompt is a span-extraction instruction; ours must read
            # like something a user would actually type.
            assert "Highlight the parts" not in question
            assert question.endswith("?")
