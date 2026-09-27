"""Provider-agnostic LLM layer (Shared Module 2: LLM APIs & Model Providers).

Everything the rest of the app needs from a model lives behind `LLMClient`:
plain completion, streaming, structured output (JSON mode), tool calling, and
token counting. Anthropic Claude is the implementation that ships; the abstract
base means a second provider is a new subclass, not a refactor.

Claude-specific details worth knowing if you read this later:
  * Opus 5 runs adaptive thinking by default; `budget_tokens` is rejected (400).
    Depth is controlled with `output_config={"effort": ...}` instead.
  * Assistant prefill is not supported - output shape is forced with
    structured outputs, never by pre-seeding an assistant turn.
  * Prompt caching is a *prefix* match, so the system prompt must be byte
    stable. Volatile text (the question, the retrieved chunks) goes last.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence, Type, TypeVar

import anthropic
from pydantic import BaseModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens, mirroring current Anthropic list pricing. Cache reads bill
# at ~0.1x input and cache writes at ~1.25x input.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(slots=True)
class Usage:
    """Token accounting for one model call (Tier 1 M5: Cost Tracking)."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        price = PRICING.get(self.model)
        if price is None:
            return 0.0
        return (
            self.input_tokens * price["input"]
            + self.cache_read_tokens * price["input"] * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * price["input"] * CACHE_WRITE_MULTIPLIER
            + self.output_tokens * price["output"]
        ) / 1_000_000

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass(slots=True)
class LLMResult:
    text: str
    usage: Usage
    stop_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None


class LLMClient(ABC):
    """The contract every provider adapter implements."""

    @abstractmethod
    async def complete(self, *, system: str, messages: list[dict[str, Any]], **kw) -> LLMResult: ...

    @abstractmethod
    def stream(self, *, system: str, messages: list[dict[str, Any]], **kw) -> AsyncIterator[dict[str, Any]]: ...

    @abstractmethod
    async def structured(self, *, system: str, messages: list[dict[str, Any]], schema: Type[T], **kw) -> tuple[T, Usage]: ...

    @abstractmethod
    async def count_tokens(self, *, system: str, messages: list[dict[str, Any]], model: str | None = None) -> int: ...


class AnthropicClient(LLMClient):
    """Claude implementation of `LLMClient`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        kwargs: dict[str, Any] = {"max_retries": 3, "timeout": 120.0}
        if self.settings.anthropic_api_key:
            kwargs["api_key"] = self.settings.anthropic_api_key
        # Helicone (Module 8: Observability) is a reverse proxy - point the SDK
        # at it and every call is logged, with zero changes at the call sites.
        if self.settings.helicone_api_key:
            kwargs["base_url"] = "https://anthropic.helicone.ai"
            kwargs["default_headers"] = {
                "Helicone-Auth": f"Bearer {self.settings.helicone_api_key}",
                "Helicone-Property-App": self.settings.app_name,
            }
        self._client = anthropic.AsyncAnthropic(**kwargs)

    # ------------------------------------------------------------ helpers
    def _system_blocks(self, system: str, cacheable: bool) -> list[dict[str, Any]]:
        """Wrap the system prompt, marking it as a cache breakpoint.

        The system prompt is the stable prefix of every request in this app, so
        it is exactly what should be cached: ~90% cheaper on read and it never
        varies per user question.
        """
        block: dict[str, Any] = {"type": "text", "text": system}
        if cacheable and self.settings.enable_prompt_cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    @staticmethod
    def _usage_from(response: Any, model: str, latency_ms: float) -> Usage:
        u = getattr(response, "usage", None)
        return Usage(
            model=model,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            latency_ms=latency_ms,
        )

    def _request_kwargs(
        self,
        *,
        model: str,
        max_tokens: int,
        effort: str | None,
        thinking: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens}
        output_config: dict[str, Any] = {}
        if effort:
            output_config["effort"] = effort
        if output_config:
            kwargs["output_config"] = output_config
        if thinking:
            # Adaptive thinking: Claude decides how much to reason. Summarized
            # display lets the UI show a "reasoning" trace without exposing raw
            # chain of thought (which the API never returns anyway).
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        return kwargs

    # ----------------------------------------------------------- complete
    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
        thinking: bool = False,
        tools: Sequence[dict[str, Any]] | None = None,
        cacheable_system: bool = True,
    ) -> LLMResult:
        model = model or self.settings.answer_model
        max_tokens = max_tokens or self.settings.answer_max_tokens
        kwargs = self._request_kwargs(
            model=model, max_tokens=max_tokens, effort=effort, thinking=thinking
        )
        if tools:
            kwargs["tools"] = list(tools)

        started = asyncio.get_event_loop().time()
        response = await self._client.messages.create(
            system=self._system_blocks(system, cacheable_system),
            messages=messages,
            **kwargs,
        )
        latency_ms = (asyncio.get_event_loop().time() - started) * 1000

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        return LLMResult(
            text="".join(text_parts).strip(),
            usage=self._usage_from(response, model, latency_ms),
            stop_reason=response.stop_reason,
            tool_calls=tool_calls,
            raw=response,
        )

    # ------------------------------------------------------------- stream
    async def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
        thinking: bool = False,
        cacheable_system: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield `{type, ...}` events: `thinking`, `token`, and a final `usage`.

        Streaming is what makes the UI feel instant - the user sees the first
        sentence while Claude is still writing the citation list.
        """
        model = model or self.settings.answer_model
        max_tokens = max_tokens or self.settings.answer_max_tokens
        kwargs = self._request_kwargs(
            model=model, max_tokens=max_tokens, effort=effort, thinking=thinking
        )

        started = asyncio.get_event_loop().time()
        async with self._client.messages.stream(
            system=self._system_blocks(system, cacheable_system),
            messages=messages,
            **kwargs,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield {"type": "token", "text": event.delta.text}
                    elif event.delta.type == "thinking_delta":
                        yield {"type": "thinking", "text": event.delta.thinking}
            final = await stream.get_final_message()

        latency_ms = (asyncio.get_event_loop().time() - started) * 1000
        usage = self._usage_from(final, model, latency_ms)
        yield {"type": "usage", "usage": usage.as_dict(), "stop_reason": final.stop_reason}

    # --------------------------------------------------------- structured
    async def structured(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        schema: Type[T],
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
        cacheable_system: bool = True,
    ) -> tuple[T, Usage]:
        """Structured output / JSON mode (Module 2).

        `messages.parse()` constrains generation to the Pydantic schema, so the
        router, the grader and the judge cannot return prose we then have to
        regex. This is what makes the LangGraph edges safe to branch on.
        """
        model = model or self.settings.fast_model
        max_tokens = max_tokens or self.settings.fast_max_tokens
        kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens}
        if effort:
            kwargs["output_config"] = {"effort": effort}

        started = asyncio.get_event_loop().time()
        response = await self._client.messages.parse(
            system=self._system_blocks(system, cacheable_system),
            messages=messages,
            output_format=schema,
            **kwargs,
        )
        latency_ms = (asyncio.get_event_loop().time() - started) * 1000
        parsed = response.parsed_output
        if parsed is None:  # schema-constrained generation hit max_tokens
            text = next((b.text for b in response.content if b.type == "text"), "{}")
            parsed = schema.model_validate(json.loads(text))
        return parsed, self._usage_from(response, model, latency_ms)

    # -------------------------------------------------------- token count
    async def count_tokens(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> int:
        """Exact pre-flight token count (never tiktoken - that is OpenAI's
        tokenizer and undercounts Claude by 15-20%)."""
        model = model or self.settings.answer_model
        result = await self._client.messages.count_tokens(
            model=model,
            system=self._system_blocks(system, False),
            messages=messages,
        )
        return result.input_tokens


_client_singleton: LLMClient | None = None


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Factory keyed on `LLM_PROVIDER`. Adding OpenAI/Gemini = one more branch."""
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        _client_singleton = AnthropicClient(settings)
    elif settings.llm_provider == "openai":
        # Imported lazily so the `openai` package is only required when it is
        # actually the selected provider.
        from app.core.llm_openai import OpenAIClient

        _client_singleton = OpenAIClient(settings)
    elif settings.llm_provider == "ollama":
        from app.core.llm_ollama import OllamaClient

        _client_singleton = OllamaClient(settings)
    elif settings.llm_provider == "google":
        from app.core.llm_gemini import GeminiClient

        _client_singleton = GeminiClient(settings)
    else:
        raise NotImplementedError(
            f"Provider '{settings.llm_provider}' has no adapter yet. "
            "Subclass LLMClient and register it here - see llm_openai.py "
            "or llm_ollama.py."
        )
    return _client_singleton


def reset_llm_client() -> None:
    """Test hook - drops the cached singleton."""
    global _client_singleton
    _client_singleton = None
