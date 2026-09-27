"""Rate limiting, cost accounting, sessions and memory."""
from __future__ import annotations

import time

import pytest

from app.core.cost import CostTracker
from app.core.llm import PRICING, Usage
from app.core.rate_limit import CostCircuitBreaker, TokenBucket
from app.memory.session import SessionStore


class TestTokenBucket:
    def test_allows_up_to_the_burst(self):
        bucket = TokenBucket(rate_per_second=1.0, burst=5)
        assert all(bucket.consume("user")[0] for _ in range(5))

    def test_blocks_past_the_burst(self):
        bucket = TokenBucket(rate_per_second=1.0, burst=3)
        for _ in range(3):
            bucket.consume("user")
        allowed, retry_after = bucket.consume("user")
        assert allowed is False
        assert retry_after > 0

    def test_refills_over_time(self):
        bucket = TokenBucket(rate_per_second=100.0, burst=2)
        bucket.consume("user")
        bucket.consume("user")
        assert bucket.consume("user")[0] is False
        time.sleep(0.05)  # 100/s => 5 tokens back
        assert bucket.consume("user")[0] is True

    def test_clients_are_isolated(self):
        """One noisy client must not rate-limit everyone else."""
        bucket = TokenBucket(rate_per_second=1.0, burst=2)
        bucket.consume("alice")
        bucket.consume("alice")
        assert bucket.consume("alice")[0] is False
        assert bucket.consume("bob")[0] is True

    def test_sweep_drops_idle_buckets(self):
        bucket = TokenBucket(rate_per_second=1.0, burst=2)
        bucket.consume("user")
        assert bucket.sweep(max_idle_seconds=0) == 1


class TestCostCircuitBreaker:
    def test_allows_under_the_ceiling(self):
        assert CostCircuitBreaker(10.0).check(4.5)[0] is True

    def test_trips_at_the_ceiling(self):
        allowed, message = CostCircuitBreaker(10.0).check(10.0)
        assert allowed is False
        assert "budget" in message.lower()

    def test_zero_ceiling_disables_the_breaker(self):
        assert CostCircuitBreaker(0.0).check(9999.0)[0] is True


class TestUsageCosting:
    def test_prices_a_call(self):
        usage = Usage(model="claude-opus-5", input_tokens=1_000_000, output_tokens=0)
        assert usage.cost_usd == pytest.approx(PRICING["claude-opus-5"]["input"])

    def test_cached_input_is_much_cheaper(self):
        """Caching is the main cost lever, so the discount must be modelled."""
        fresh = Usage(model="claude-opus-5", input_tokens=100_000)
        cached = Usage(model="claude-opus-5", cache_read_tokens=100_000)
        assert cached.cost_usd == pytest.approx(fresh.cost_usd * 0.1)

    def test_cache_writes_cost_a_premium(self):
        fresh = Usage(model="claude-opus-5", input_tokens=100_000)
        written = Usage(model="claude-opus-5", cache_write_tokens=100_000)
        assert written.cost_usd > fresh.cost_usd

    def test_unknown_model_does_not_crash(self):
        assert Usage(model="some-future-model", input_tokens=1000).cost_usd == 0.0

    def test_total_tokens_includes_cache(self):
        usage = Usage(model="claude-haiku-4-5", input_tokens=10, output_tokens=5,
                      cache_read_tokens=3, cache_write_tokens=2)
        assert usage.total_tokens == 20


class TestCostTracker:
    def test_records_and_aggregates_by_stage(self, tmp_path):
        tracker = CostTracker(tmp_path / "cost.db")
        tracker.record(Usage(model="claude-opus-5", input_tokens=1000, output_tokens=500),
                       stage="graph.generate", session_id="s1")
        tracker.record(Usage(model="claude-haiku-4-5", input_tokens=200, output_tokens=50),
                       stage="graph.route", session_id="s1")

        summary = tracker.summary(days=1)
        assert summary.calls == 2
        assert "graph.generate" in summary.by_stage
        assert summary.total_cost_usd > 0
        assert tracker.spent_today() > 0

    def test_cache_hit_rate(self, tmp_path):
        tracker = CostTracker(tmp_path / "cost2.db")
        tracker.record(Usage(model="claude-opus-5", input_tokens=100, cache_read_tokens=900),
                       stage="graph.generate")
        assert tracker.summary(days=1).cache_hit_rate == pytest.approx(0.9)

    def test_ignores_calls_with_no_model(self, tmp_path):
        tracker = CostTracker(tmp_path / "cost3.db")
        assert tracker.record(Usage(), stage="noop") == 0.0
        assert tracker.summary(days=1).calls == 0


class TestSessionStore:
    @pytest.fixture
    def store(self, tmp_path):
        return SessionStore(tmp_path / "sessions.db", ttl_seconds=3600)

    def test_create_and_read_back(self, store):
        session = store.create(title="Distributor review")
        loaded = store.get(session.session_id)
        assert loaded is not None
        assert loaded.title == "Distributor review"

    def test_messages_persist_in_order(self, store):
        session = store.create()
        store.append_message(session.session_id, "user", "What is the governing law?")
        store.append_message(session.session_id, "assistant", "Delaware [1].")
        loaded = store.get(session.session_id)
        assert [m.role for m in loaded.messages] == ["user", "assistant"]

    def test_expired_sessions_are_not_returned(self, tmp_path):
        store = SessionStore(tmp_path / "expiring.db", ttl_seconds=-1)
        session = store.create()
        assert store.get(session.session_id) is None

    def test_activity_extends_the_ttl(self, store):
        """A live conversation must not expire mid-thread."""
        session = store.create()
        original = store.get(session.session_id).expires_at
        time.sleep(0.01)
        store.append_message(session.session_id, "user", "follow-up")
        assert store.get(session.session_id).expires_at > original

    def test_get_or_create_handles_a_stale_id(self, store):
        session = store.get_or_create("does-not-exist")
        assert session.session_id != "does-not-exist"

    def test_delete_removes_messages_too(self, store):
        session = store.create()
        store.append_message(session.session_id, "user", "hello")
        store.delete(session.session_id)
        assert store.get(session.session_id) is None

    def test_filters_round_trip(self, store):
        session = store.create()
        store.set_filters(session.session_id, {"doc_ids": ["doc1"], "clause_types": ["insurance"]})
        assert store.get(session.session_id).filters["doc_ids"] == ["doc1"]

    def test_title_is_only_set_once(self, store):
        """The title comes from the opening question and should not churn."""
        session = store.create()
        store.set_title(session.session_id, "First question")
        store.set_title(session.session_id, "Second question")
        assert store.get(session.session_id).title == "First question"


class TestConversationMemory:
    def test_history_starts_with_a_user_turn(self, tmp_path, settings):
        """The Messages API rejects a history that starts with an assistant turn."""
        from app.memory.chat_memory import ConversationMemory
        from app.memory import session as session_module

        store = SessionStore(tmp_path / "mem.db", ttl_seconds=3600)
        session_module._store = store
        try:
            memory = ConversationMemory(settings)
            session = store.create()
            # Simulate a session whose first stored row is an assistant message.
            store.append_message(session.session_id, "assistant", "Orphaned answer")
            store.append_message(session.session_id, "user", "Real question")
            loaded = store.get(session.session_id)
            messages = memory.as_api_messages(loaded)
            assert messages and messages[0]["role"] == "user"
        finally:
            session_module._store = None

    def test_derives_a_title_from_the_question(self, settings):
        from app.memory.chat_memory import ConversationMemory

        memory = ConversationMemory(settings)
        title = memory.derive_title("What   are the termination rights?  ")
        assert title == "What are the termination rights?"

    def test_long_titles_are_truncated(self, settings):
        from app.memory.chat_memory import ConversationMemory

        memory = ConversationMemory(settings)
        assert len(memory.derive_title("x" * 300)) <= 81
