"""Google Gemini adapter for `LLMClient` — the free-tier cloud option.

Fourth provider behind the same interface. Gemini earns its place here because
its free tier needs no credit card and is generous enough to actually run this
application, which makes it the realistic choice for someone who wants the
system working today without a billing account.

Provider differences handled here:

  * **System prompt is a config field, not a message.** Gemini takes it as
    `system_instruction` on `GenerateContentConfig` rather than a turn in the
    conversation.
  * **Roles are `user` / `model`,** not `user` / `assistant`, and the payload
    shape is `{"role": ..., "parts": [{"text": ...}]}`.
  * **Native structured output.** `response_mime_type="application/json"` plus
    `response_schema` constrains generation to a Pydantic model, so the router
    and graders get validated objects exactly as on the other providers.
  * **Free tier means rate limits, not bills.** Pricing is recorded for the paid
    tier so the cost page stays meaningful, but the models listed as free bill
    at zero.
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

# USD per 1M tokens on the paid tier. The free tier bills nothing; these exist
# so the cost page still reports something meaningful if billing is enabled.
PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-pro":        {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash":      {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash":      {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
}


def price_for(model: str) -> dict[str, float] | None:
    if model in PRICING:
        return PRICING[model]
    candidates = [k for k in PRICING if model.startswith(k)]
    return PRICING[max(candidates, key=len)] if candidates else None


class GeminiUsage(Usage):
    @property
    def cost_usd(self) -> float:
        price = price_for(self.model)
        if price is None:
            return 0.0
        return (
            self.input_tokens * price["input"] + self.output_tokens * price["output"]
        ) / 1_000_000


class GeminiClient(LLMClient):
    def __init__(self, settings: Settings | None = None) -> None:
        from google import genai

        self.settings = settings or get_settings()
        if not self.settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set but LLM_PROVIDER=google.")
        self._client = genai.Client(api_key=self.settings.google_api_key)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Our message list -> Gemini's `contents`.

        Two conversions: the assistant role is called `model`, and text lives in
        a `parts` list rather than a bare string.
        """
        out: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            role = "model" if message.get("role") == "assistant" else "user"
            out.append({"role": role, "parts": [{"text": content}]})
        return out

    def _config(self, *, system: str, max_tokens: int, **extra: Any) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=self.settings.gemini_temperature,
            **extra,
        )

    @staticmethod
    def _usage_from(response: Any, model: str, latency_ms: float) -> Usage:
        meta = getattr(response, "usage_metadata", None)
        return GeminiUsage(
            model=model,
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            cache_read_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
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
        model = model or self.settings.gemini_answer_model
        max_tokens = max_tokens or self.settings.answer_max_tokens

        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = [{"function_declarations": [_to_gemini_tool(t) for t in tools]}]

        started = asyncio.get_event_loop().time()
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=self._contents(messages),
            config=self._config(system=system, max_tokens=max_tokens, **extra),
        )
        latency_ms = (asyncio.get_event_loop().time() - started) * 1000

        tool_calls: list[dict[str, Any]] = []
        for i, call in enumerate(getattr(response, "function_calls", None) or []):
            tool_calls.append({"id": f"call_{i}", "name": call.name, "input": dict(call.args or {})})

        return LLMResult(
            text=(response.text or "").strip(),
            usage=self._usage_from(response, model, latency_ms),
            stop_reason="tool_use" if tool_calls else "end_turn",
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
        model = model or self.settings.gemini_answer_model
        max_tokens = max_tokens or self.settings.answer_max_tokens

        started = asyncio.get_event_loop().time()
        final: Any = None

        stream = await self._client.aio.models.generate_content_stream(
            model=model,
            contents=self._contents(messages),
            config=self._config(system=system, max_tokens=max_tokens),
        )
        async for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield {"type": "token", "text": text}
            # Usage only arrives on the last chunk that carries it.
            if getattr(chunk, "usage_metadata", None):
                final = chunk

        latency_ms = (asyncio.get_event_loop().time() - started) * 1000
        usage = self._usage_from(final, model, latency_ms) if final else GeminiUsage(model=model)
        yield {"type": "usage", "usage": usage.as_dict(), "stop_reason": "end_turn"}

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
        model = model or self.settings.gemini_fast_model
        max_tokens = max_tokens or self.settings.fast_max_tokens

        started = asyncio.get_event_loop().time()
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=self._contents(messages),
            config=self._config(
                system=system,
                max_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        latency_ms = (asyncio.get_event_loop().time() - started) * 1000

        # The SDK parses into the schema when it can; fall back to the raw JSON.
        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, schema):
            parsed = schema.model_validate(json.loads(response.text or "{}"))
        return parsed, self._usage_from(response, model, latency_ms)

    # -------------------------------------------------------- token count
    async def count_tokens(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> int:
        model = model or self.settings.gemini_answer_model
        result = await self._client.aio.models.count_tokens(
            model=model, contents=self._contents(messages)
        )
        return int(getattr(result, "total_tokens", 0) or 0)


def _to_gemini_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Anthropic tool shape -> Gemini function declaration."""
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
    }
