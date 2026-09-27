"""Retrieval maths: fusion, diversity, sparse matching, and the metrics."""
from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.metrics import citation_accuracy, score_retrieval
from app.retrieval.bm25 import BM25Index, _matches, tokenize
from app.retrieval.hybrid import maximal_marginal_relevance, reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk


def chunk(cid: str, text: str = "text", **meta) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, doc_id=meta.pop("doc_id", "doc1"), text=text, metadata=meta)


class TestReciprocalRankFusion:
    def test_agreement_between_arms_wins(self):
        """A chunk both retrievers like should beat one only a single arm found."""
        dense = [chunk("a"), chunk("b"), chunk("c")]
        sparse = [chunk("c"), chunk("a"), chunk("d")]
        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        assert fused[0].chunk_id == "a"  # ranked 1st and 2nd
        assert {c.chunk_id for c in fused} == {"a", "b", "c", "d"}

    def test_scores_descend(self):
        fused = reciprocal_rank_fusion([chunk("a"), chunk("b")], [chunk("b"), chunk("c")], k=60)
        scores = [c.fused_score for c in fused]
        assert scores == sorted(scores, reverse=True)

    def test_marks_chunks_found_by_both_arms(self):
        fused = reciprocal_rank_fusion([chunk("a")], [chunk("a")], k=60)
        assert fused[0].retriever == "hybrid"

    def test_single_arm_still_works(self):
        fused = reciprocal_rank_fusion([chunk("a"), chunk("b")], [], k=60)
        assert [c.chunk_id for c in fused] == ["a", "b"]

    def test_weights_shift_the_ranking(self):
        dense = [chunk("d1"), chunk("d2")]
        sparse = [chunk("s1"), chunk("s2")]
        dense_led = reciprocal_rank_fusion(dense, sparse, dense_weight=0.9, sparse_weight=0.1)
        sparse_led = reciprocal_rank_fusion(dense, sparse, dense_weight=0.1, sparse_weight=0.9)
        assert dense_led[0].chunk_id == "d1"
        assert sparse_led[0].chunk_id == "s1"


class TestMMR:
    def test_breaks_up_near_duplicates(self):
        """Three near-identical chunks plus one distinct one.

        The first pick is always pure relevance (c1 is marginally closest to the
        query). The second pick is where MMR earns its place: at a
        diversity-leaning lambda it reaches for the distinct c3 instead of
        returning a third copy of the same clause.
        """
        query = [1.0, 0.0]
        vectors = [[1.0, 0.02], [1.0, 0.01], [1.0, 0.03], [0.6, 0.8]]
        chunks = [chunk(f"c{i}") for i in range(4)]
        selected = maximal_marginal_relevance(chunks, query, vectors, k=2, lambda_mult=0.2)
        assert selected[0].chunk_id == "c1"
        assert selected[1].chunk_id == "c3"

    def test_relevance_still_wins_at_high_lambda(self):
        """The same pool at a relevance-leaning lambda keeps the near-duplicate,
        which is the correct behaviour - lambda is the dial between the two."""
        query = [1.0, 0.0]
        vectors = [[1.0, 0.02], [1.0, 0.01], [1.0, 0.03], [0.6, 0.8]]
        chunks = [chunk(f"c{i}") for i in range(4)]
        selected = maximal_marginal_relevance(chunks, query, vectors, k=2, lambda_mult=0.9)
        assert selected[1].chunk_id != "c3"

    def test_lambda_one_is_pure_relevance(self):
        query = [1.0, 0.0]
        vectors = [[0.9, 0.1], [1.0, 0.0], [0.2, 0.9]]
        chunks = [chunk(f"c{i}") for i in range(3)]
        selected = maximal_marginal_relevance(chunks, query, vectors, k=2, lambda_mult=1.0)
        assert selected[0].chunk_id == "c1"

    def test_returns_everything_when_k_exceeds_pool(self):
        chunks = [chunk("a"), chunk("b")]
        selected = maximal_marginal_relevance(chunks, [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], k=5)
        assert len(selected) == 2

    def test_mismatched_vectors_degrade_gracefully(self):
        """A vector/chunk length mismatch must not crash the request."""
        chunks = [chunk("a"), chunk("b"), chunk("c")]
        selected = maximal_marginal_relevance(chunks, [1.0, 0.0], [[1.0, 0.0]], k=2)
        assert len(selected) == 2


class TestBM25:
    def test_keeps_legal_identifiers_intact(self):
        tokens = tokenize("Payment of $5,000,000 under Section 12.3 within 30 days")
        assert "$5,000,000" in tokens
        assert "12.3" in tokens

    def test_drops_stopwords(self):
        assert "the" not in tokenize("the agreement and the party")

    def test_finds_exact_terms_dense_search_blurs(self):
        index = BM25Index()
        index.build([
            chunk("a", "Siemens Aktiengesellschaft shall distribute the Products."),
            chunk("b", "The supplier shall deliver goods to the customer."),
            chunk("c", "Termination requires ninety days notice."),
        ])
        results = index.search("Siemens", k=3)
        assert results and results[0].chunk_id == "a"

    def test_empty_index_returns_nothing(self):
        assert BM25Index().search("anything", k=5) == []

    def test_respects_metadata_filters(self):
        index = BM25Index()
        index.build([
            chunk("a", "termination for convenience", doc_id="doc1"),
            chunk("b", "termination for convenience", doc_id="doc2"),
        ])
        results = index.search("termination", k=5, where={"doc_id": "doc1"})
        assert [r.chunk_id for r in results] == ["a"]


class TestFilterMatching:
    @pytest.mark.parametrize(
        "meta,where,expected",
        [
            ({"doc_id": "a"}, None, True),
            ({"doc_id": "a"}, {"doc_id": "a"}, True),
            ({"doc_id": "a"}, {"doc_id": "b"}, False),
            ({"doc_id": "a"}, {"doc_id": {"$in": ["a", "b"]}}, True),
            ({"clause_types_csv": ",insurance,audit_rights,"}, {"clause_types_csv": {"$contains": ",insurance,"}}, True),
            ({"clause_types_csv": ",insurance,"}, {"clause_types_csv": {"$contains": ",non_compete,"}}, False),
            ({"doc_id": "a"}, {"$or": [{"doc_id": "a"}, {"doc_id": "z"}]}, True),
            ({"doc_id": "a", "governing_law": "Delaware"}, {"$and": [{"doc_id": "a"}, {"governing_law": "Delaware"}]}, True),
        ],
    )
    def test_where_clause_grammar(self, meta, where, expected):
        assert _matches(meta, where) is expected

    def test_comma_wrapping_prevents_prefix_collisions(self):
        """",non_compete," must not match ",non_competition,"."""
        meta = {"clause_types_csv": ",non_competition,"}
        assert _matches(meta, {"clause_types_csv": {"$contains": ",non_compete,"}}) is False


class TestRetrievalMetrics:
    def test_all_metrics_stay_within_range(self):
        """One gold span spread over several chunks used to push recall above 1."""
        scores = score_retrieval([True, True, True, False], spans_found=1, spans_total=1)
        for value in (scores.hit_rate, scores.span_recall, scores.precision, scores.mrr, scores.ndcg):
            assert 0.0 <= value <= 1.0

    def test_span_recall_counts_spans_not_chunks(self):
        scores = score_retrieval([True, True], spans_found=1, spans_total=2)
        assert scores.span_recall == 0.5
        assert scores.relevant_chunks == 2

    def test_mrr_rewards_higher_placement(self):
        first = score_retrieval([True, False, False], 1, 1)
        third = score_retrieval([False, False, True], 1, 1)
        assert first.mrr == 1.0
        assert third.mrr == pytest.approx(1 / 3)
        assert first.ndcg > third.ndcg

    def test_nothing_relevant_scores_zero(self):
        scores = score_retrieval([False, False], spans_found=0, spans_total=2)
        assert scores.hit_rate == 0.0
        assert scores.ndcg == 0.0
        assert scores.first_relevant_rank is None

    def test_perfect_ranking_scores_one(self):
        scores = score_retrieval([True, True, False, False], spans_found=2, spans_total=2)
        assert scores.ndcg == pytest.approx(1.0)
        assert scores.span_recall == 1.0

    def test_empty_result_list(self):
        scores = score_retrieval([], spans_found=0, spans_total=3)
        assert scores.hit_rate == 0.0
        assert scores.precision == 0.0


class TestCitationAccuracy:
    def test_scores_only_cited_excerpts(self):
        citations = [
            {"index": 1, "is_relevant": True},
            {"index": 2, "is_relevant": False},
            {"index": 3, "is_relevant": True},
        ]
        result = citation_accuracy("The cap is stated in [1] and confirmed by [3].", citations)
        assert result["cited_count"] == 2
        assert result["citation_precision"] == 1.0

    def test_flags_citations_that_do_not_exist(self):
        result = citation_accuracy("As set out in [7].", [{"index": 1, "is_relevant": True}])
        assert result["invalid_citations"] == [7]

    def test_detects_an_uncited_answer(self):
        result = citation_accuracy("The cap is one million dollars.", [{"index": 1, "is_relevant": True}])
        assert result["uncited"] is True
