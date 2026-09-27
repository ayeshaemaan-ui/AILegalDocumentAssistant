"""OpenAI adapter for `LLMClient` (Shared Module 2: LLM APIs & Model Providers).

This file exists to prove the abstraction was real. `LLMClient` was written as
an ABC so that adding a provider is a subclass rather than a refactor, and this
is that subclass: nothing in the graph, the retrieval pipeline, the evaluation
harness or the API changed to support it. Switching providers is one env var.

Provider differences that actually matter, handled here:

  * **Parameter name.** The gpt-5 and o-series models reject `max_tokens` and
    require `max_completion_tokens`. The 4.x family accepts the old name.
  * **Reasoning models ignore sampling.** gpt-5 / o-series reject a custom
    `temperature`, so it is only sent for models that accept it.
  * **System role.** Reasoning models prefer a `developer` role message over
    `system`; both are accepted, but `developer` is the documented one.
  * **Caching is automatic.** There is no `cache_control` to place - OpenAI
    caches long prefixes on its own and reports hits in
    `usage.prompt_tokens_details.cached_tokens`. Keeping the system prompt
    byte-stable still matters; we just do not annotate it.
  * **Token counting is local.** There is no count-tokens endpoint, so tiktoken
    is used - which is correct *here* and would be wrong for Claude, whose
    tokeniser it does not model.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Sequence, Type, TypeVar

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.core.llm import LLMClient, LLMResult, Usage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (OpenAI list pricing at time of writing). Unknown models
# cost 0.0 rather than raising - a pricing gap must never break a request.
PRICING: dict[str, dict[str, float]] = {
    "gpt-5":          {"input": 1.25, "output": 10.00, "cached": 0.125},
    "gpt-5-mini":     {"input": 0.25, "output": 2.00,  "cached": 0.025},
    "gpt-5-nano":     {"input": 0.05, "output": 0.40,  "cached": 0.005},
    "gpt-4.1":        {"input": 2.00, "output": 8.00,  "cached": 0.50},
    "gpt-4.1-mini":   {"input": 0.40, "output": 1.60,  "cached": 0.10},
    "gpt-4.1-nano":   {"input": 0.10, "output": 0.40,  "cached": 0.025},
    "gpt-4o":         {"input": 2.50, "output": 10.00, "cached": 1.25},
    "gpt-4o-mini":    {"input": 0.15, "output": 0.60,  "cached": 0.075},
    "o3":             {"input": 2.00, "output": 8.00,  "cached": 0.50},
    "o4-mini":        {"input": 1.10, "output": 4.40,  "cached": 0.275},
}

# Families that reject `max_tokens` and `temperature`.
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning(model: str) -> bool:
    return model.startswith(REASONING_PREFIXES)


def price_for(model: str) -> dict[str, float] | None:
    """Exact match first, then longest prefix - so `gpt-5-mini-2025-08-07`
    prices as `gpt-5-mini` and not as `gpt-5`."""
    if model in PRICING:
        return PRICING[model]
    candidates = [k for k in PRICING if model.startswith(k)]
    return PRICING[max(candidates, key=len)] if candidates else None


class OpenAIUsage(Usage):
    """`Usage` with OpenAI's cache pricing, which differs per family.

    Anthropic bills cache reads at a flat ~0.1x of input. OpenAI's discount is
    50% on the 4.x models and 90% on gpt-5, so the rate has to come from the
    pricing table rather than a constant.
    """

    @property
    def cost_usd(self) -> float:
        price = price_for(self.model)
        if price is None:
            return 0.0
        return (
            self.input_tokens * price["input"]
            + self.cache_read_tokens * price.get("cached", price["input"])
            + self.output_tokens * price["output"]
        ) / 1_000_000


class OpenAIClient(LLMClient):
    def __init__(self, settings: Settings | None = None) -> None:
        from openai import AsyncOpenAI

        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai.")
        self._client = AsyncOpenAI(
            api_key=self.settings.openai_api_key, max_retries=3, timeout=120.0
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _messages(system: str, messages: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
        role = "developer" if _is_reasoning(model) else "system"
        return [{"role": role, "content": system}, *messages]

    @staticmethod
    def _token_kwargs(model: str, max_tokens: int) -> dict[str, Any]:
        key = "max_completion_tokens" if _is_reasoning(model) else "max_tokens"
        return {key: max_tokens}

    @staticmethod
    def _usage_from(response: Any, model: str, latency_ms: float) -> Usage:
        u = getattr(response, "usage", None)
        prompt = getattr(u, "prompt_tokens", 0) or 0
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        return OpenAIUsage(
            model=model,
            # OpenAI reports cached tokens *inside* prompt_tokens, so the
            # uncached figure is the difference. Passing prompt_tokens straight
            # through would double-count and overstate the bill.
            input_tokens=max(0, prompt - cached),
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            cache_read_tokens=cached,
            cache_write_tokens=0,  # caching is automatic; writes are not billed
            latency_ms=latency_ms,
        )

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
        model = model or self.settings.openai_answer_model
        max_tokens = max_tokens or self.settings.answer_max_tokens

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._messages(system, messages, model),
            **self._token_kwargs(model, max_tokens),
        }
        if effort and _is_reasoning(model):
            # OpenAI's scale has no "xhigh"/"max"; fold those onto "high".
            kwargs["reasoning_effort"] = effort if effort in {"low", "medium", "high"} else "high"
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]

        started = asyncio.get_event_loop().time()
        response = await self._client.chat.completions.create(**kwargs)
        latency_ms = (asyncio.get_event_loop().time() - started) * 1000

        choice = response.choices[0]
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "input": json.loads(tc.function.arguments or "{}")}
            for tc in (choice.message.tool_calls or [])
        ]
        return LLMResult(
            text=(choice.message.content or "").strip(),
            usage=self._usage_from(response, model, latency_ms),
            stop_reason=choice.finish_reason,
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
        model = model or self.settings.openai_answer_model
        max_tokens = max_tokens or self.settings.answer_max_tokens

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._messages(system, messages, model),
            "stream": True,
            # Without this the final chunk carries no usage and cost tracking
            # silently records zero for every streamed answer.
            "stream_options": {"include_usage": True},
            **self._token_kwargs(model, max_tokens),
        }
        if effort and _is_reasoning(model):
            kwargs["reasoning_effort"] = effort if effort in {"low", "medium", "high"} else "high"

        started = asyncio.get_event_loop().time()
        final_usage: Any = None
        finish_reason: str | None = None

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.usage is not None:
                final_usage = chunk.usage
            if not chunk.choices:
                continue  # the usage-only terminal chunk has an empty choices list
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield {"type": "token", "text": delta.content}
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        latency_ms = (asyncio.get_event_loop().time() - started) * 1000
        usage = self._usage_from(
            type("R", (), {"usage": final_usage})(), model, latency_ms
        )
        yield {"type": "usage", "usage": usage.as_dict(), "stop_reason": finish_reason}

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
        model = model or self.settings.openai_fast_model
        max_tokens = max_tokens or self.settings.fast_max_tokens

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._messages(system, messages, model),
            "response_format": schema,
            **self._token_kwargs(model, max_tokens),
        }

        started = asyncio.get_event_loop().time()
        response = await self._client.chat.completions.parse(**kwargs)
        latency_ms = (asyncio.get_event_loop().time() - started) * 1000

        parsed = response.choices[0].message.parsed
        if parsed is None:  # hit the token ceiling mid-object
            content = response.choices[0].message.content or "{}"
            parsed = schema.model_validate(json.loads(content))
        return parsed, self._usage_from(response, model, latency_ms)

    # -------------------------------------------------------- token count
    async def count_tokens(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> int:
        """Local count via tiktoken - correct for OpenAI models specifically.

        (The same library would be wrong for Claude, which is why the Anthropic
        adapter calls the count-tokens endpoint instead of importing this.)
        """
        model = model or self.settings.openai_answer_model

        def _count() -> int:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("o200k_base")
            total = len(encoding.encode(system))
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    total += len(encoding.encode(content))
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            total += len(encoding.encode(block["text"]))
                total += 4  # per-message framing overhead
            return total

        return await asyncio.to_thread(_count)


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Anthropic tool shape -> OpenAI tool shape.

    Tools are defined once in `agents/tools.py` in Anthropic's format; this
    translates rather than duplicating the definitions.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            **({"strict": True} if tool.get("strict") else {}),
        },
    }
