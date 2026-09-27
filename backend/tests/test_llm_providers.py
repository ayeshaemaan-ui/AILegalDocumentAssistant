"""Provider abstraction: does swapping LLM_PROVIDER actually work?

These assert the shape of the adapters and the provider-specific quirks they
exist to absorb. They make no network calls - the point is that the abstraction
holds, not that a vendor's servers are up.
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.core.llm import LLMClient
from app.core.llm_openai import (
    PRICING,
    OpenAIClient,
    OpenAIUsage,
    _is_reasoning,
    _to_openai_tool,
    price_for,
)


class TestModelResolution:
    def test_anthropic_provider_resolves_claude_models(self):
        s = Settings(llm_provider="anthropic")
        assert s.resolved_answer_model.startswith("claude")
        assert s.resolved_fast_model.startswith("claude")

    def test_openai_provider_resolves_gpt_models(self):
        s = Settings(llm_provider="openai")
        assert s.resolved_answer_model.startswith("gpt")
        assert s.resolved_judge_model.startswith("gpt")

    def test_switching_provider_does_not_clobber_the_other_config(self):
        """Both sets of model names coexist, so flipping the provider back does
        not require re-entering them."""
        s = Settings(llm_provider="openai")
        assert s.answer_model.startswith("claude")
        assert s.openai_answer_model.startswith("gpt")


class TestReasoningModelQuirks:
    @pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "o3", "o4-mini", "o1"])
    def test_detects_reasoning_families(self, model):
        assert _is_reasoning(model) is True

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])
    def test_detects_non_reasoning_families(self, model):
        assert _is_reasoning(model) is False

    def test_reasoning_models_get_max_completion_tokens(self):
        """gpt-5 and the o-series reject `max_tokens` outright."""
        assert OpenAIClient._token_kwargs("gpt-5", 512) == {"max_completion_tokens": 512}

    def test_chat_models_get_max_tokens(self):
        assert OpenAIClient._token_kwargs("gpt-4o", 512) == {"max_tokens": 512}

    def test_reasoning_models_use_the_developer_role(self):
        messages = OpenAIClient._messages("SYS", [{"role": "user", "content": "hi"}], "gpt-5")
        assert messages[0]["role"] == "developer"

    def test_chat_models_use_the_system_role(self):
        messages = OpenAIClient._messages("SYS", [{"role": "user", "content": "hi"}], "gpt-4o")
        assert messages[0]["role"] == "system"

    def test_system_prompt_is_prepended_not_appended(self):
        messages = OpenAIClient._messages("SYS", [{"role": "user", "content": "hi"}], "gpt-4o")
        assert messages[0]["content"] == "SYS"
        assert messages[1]["content"] == "hi"


class TestPricing:
    def test_exact_match(self):
        assert price_for("gpt-5") == PRICING["gpt-5"]

    def test_dated_snapshot_prices_as_its_base_model(self):
        """`gpt-5-mini-2025-08-07` must price as gpt-5-mini, not as gpt-5 -
        longest-prefix wins, or the cheap model would bill at 5x."""
        assert price_for("gpt-5-mini-2025-08-07") == PRICING["gpt-5-mini"]

    def test_unknown_model_does_not_raise(self):
        assert price_for("gpt-99-imaginary") is None
        assert OpenAIUsage(model="gpt-99-imaginary", input_tokens=1000).cost_usd == 0.0

    def test_cached_input_is_discounted(self):
        fresh = OpenAIUsage(model="gpt-5", input_tokens=1_000_000)
        cached = OpenAIUsage(model="gpt-5", cache_read_tokens=1_000_000)
        assert cached.cost_usd < fresh.cost_usd
        assert cached.cost_usd == pytest.approx(PRICING["gpt-5"]["cached"])

    def test_cache_discount_differs_between_families(self):
        """OpenAI discounts 90% on gpt-5 but only 50% on gpt-4o, so the rate
        has to come from the table rather than a shared constant."""
        gpt5_ratio = PRICING["gpt-5"]["cached"] / PRICING["gpt-5"]["input"]
        gpt4o_ratio = PRICING["gpt-4o"]["cached"] / PRICING["gpt-4o"]["input"]
        assert gpt5_ratio == pytest.approx(0.1)
        assert gpt4o_ratio == pytest.approx(0.5)


class TestUsageAccounting:
    def test_cached_tokens_are_not_double_counted(self):
        """OpenAI reports cached tokens *inside* prompt_tokens. Passing the raw
        figure through would bill the same tokens twice."""
        class FakeDetails:
            cached_tokens = 800

        class FakeUsage:
            prompt_tokens = 1000
            completion_tokens = 50
            prompt_tokens_details = FakeDetails()

        class FakeResponse:
            usage = FakeUsage()

        usage = OpenAIClient._usage_from(FakeResponse(), "gpt-5", 120.0)
        assert usage.cache_read_tokens == 800
        assert usage.input_tokens == 200          # 1000 - 800, not 1000
        assert usage.total_tokens == 1050         # 200 + 800 + 50

    def test_handles_a_response_with_no_usage(self):
        class FakeResponse:
            usage = None

        usage = OpenAIClient._usage_from(FakeResponse(), "gpt-5", 10.0)
        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.0


class TestToolTranslation:
    def test_anthropic_tool_shape_converts_to_openai(self):
        """Tools are defined once in Anthropic's format; the adapter translates
        rather than the project maintaining two copies."""
        anthropic_tool = {
            "name": "search_contracts",
            "description": "Search the corpus",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        }
        converted = _to_openai_tool(anthropic_tool)
        assert converted["type"] == "function"
        assert converted["function"]["name"] == "search_contracts"
        assert converted["function"]["parameters"]["required"] == ["query"]
        assert converted["function"]["strict"] is True

    def test_tool_without_strict_omits_the_flag(self):
        converted = _to_openai_tool({"name": "t", "input_schema": {"type": "object"}})
        assert "strict" not in converted["function"]


class TestFactory:
    def test_openai_adapter_satisfies_the_interface(self):
        """The whole point of the ABC: no call site changes when the provider
        does."""
        assert issubclass(OpenAIClient, LLMClient)
        for method in ("complete", "stream", "structured", "count_tokens"):
            assert callable(getattr(OpenAIClient, method))

    def test_missing_key_fails_loudly_at_construction(self):
        s = Settings(llm_provider="openai", openai_api_key=None)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIClient(s)

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "google", "ollama"])
    def test_every_declared_provider_has_an_adapter(self, provider):
        """`llm_provider` is a closed Literal, so every value it allows must
        resolve to a real adapter - otherwise a valid config raises at runtime."""
        from app.core.llm import get_llm_client, reset_llm_client

        reset_llm_client()
        try:
            settings = Settings(
                llm_provider=provider,
                anthropic_api_key="test",
                openai_api_key="test",
                google_api_key="test",
            )
            client = get_llm_client(settings)
            assert isinstance(client, LLMClient)
        finally:
            reset_llm_client()

    def test_resolver_covers_every_provider(self):
        """A provider with no model mapping would silently fall through to the
        Claude names and call a model that does not exist on that vendor."""
        expected_prefix = {
            "anthropic": "claude", "openai": "gpt", "google": "gemini", "ollama": "qwen",
        }
        for provider, prefix in expected_prefix.items():
            s = Settings(llm_provider=provider)
            assert s.resolved_answer_model.startswith(prefix), provider
            assert s.resolved_fast_model.startswith(prefix), provider
            assert s.resolved_judge_model.startswith(prefix), provider
