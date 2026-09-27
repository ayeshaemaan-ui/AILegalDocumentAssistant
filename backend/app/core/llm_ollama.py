"""Ollama adapter for `LLMClient` — local models, zero cost, no API key.

The third provider behind the same interface. Nothing in the graph, retrieval,
evaluation or API layer knows this exists; `LLM_PROVIDER=ollama` is the whole
change.

What is genuinely different about a local model, and handled here:

  * **No hosted structured-output mode.** Ollama exposes a `format` field that
    takes a JSON Schema, which constrains generation the same way - so the
    router and graders still get validated objects rather than prose that has
    to be regexed. Where a small model still drifts, `structured()` retries once
    with the validation error fed back before giving up.
  * **Cost is genuinely zero**, so `Usage.cost_usd` is 0.0 by construction, not
    an unknown-model fallback. Token counts are still reported, because the
    cost page showing "0.00 across 47 calls" is meaningful information.
  * **Smaller context and slower generation.** A 7B model at ~4k context cannot
    take 6 long clauses plus a rolling summary, so excerpts are truncated to fit
    rather than silently overflowing.
  * **No prompt caching** to configure - the model is already local, so the
    thing caching saves (network + vendor billing) does not exist here.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.core.llm import LLMClient, LLMResult, Usage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaUsage(Usage):
    """Tokens are still counted; the bill is always zero."""

    @property
    def cost_usd(self) -> float:
        return 0.0


class OllamaClient(LLMClient):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        import httpx

        self._http = httpx.AsyncClient(
            base_url=self.settings.ollama_host.rstrip("/"),
            # Local generation on CPU/Metal is slow; a 60s default would time
            # out mid-answer on a long legal question.
            timeout=httpx.Timeout(600.0, connect=10.0),
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                # Flatten Anthropic-style content blocks into plain text.
                content = "\n".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            out.append({"role": message.get("role", "user"), "content": content})
        return out

    def _options(self, max_tokens: int) -> dict[str, Any]:
        return {
            "num_predict": max_tokens,
            "num_ctx": self.settings.ollama_context_tokens,
            "temperature": self.settings.ollama_temperature,
        }

    @staticmethod
    def _usage_from(payload: dict[str, Any], model: str, latency_ms: float) -> Usage:
        return OllamaUsage(
            model=model,
            input_tokens=payload.get("prompt_eval_count", 0) or 0,
            output_tokens=payload.get("eval_count", 0) or 0,
            latency_ms=latency_ms,
        )

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._http.post(path, json=body)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed ({type(exc).__name__}: {exc}). "
                f"Is the server running at {self.settings.ollama_host}? "
                "Start it with `ollama serve`."
            ) from exc

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
        model = model or self.settings.ollama_answer_model
        max_tokens = max_tokens or self.settings.answer_max_tokens

        body: dict[str, Any] = {
            "model": model,
            "messages": self._messages(system, messages),
            "stream": False,
            "options": self._options(max_tokens),
        }
        if tools:
            body["tools"] = [_to_ollama_tool(t) for t in tools]

        started = asyncio.get_event_loop().time()
        payload = await self._post("/api/chat", body)
        latency_ms = (asyncio.get_event_loop().time() - started) * 1000

        message = payload.get("message", {})
        tool_calls = [
            {
                "id": f"call_{i}",
                "name": tc.get("function", {}).get("name", ""),
                "input": tc.get("function", {}).get("arguments", {}) or {},
            }
            for i, tc in enumerate(message.get("tool_calls", []) or [])
        ]
        return LLMResult(
            text=(message.get("content") or "").strip(),
            usage=self._usage_from(payload, model, latency_ms),
            stop_reason=payload.get("done_reason") or "end_turn",
            tool_calls=tool_calls,
            raw=payload,
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
        model = model or self.settings.ollama_answer_model
        max_tokens = max_tokens or self.settings.answer_max_tokens

        body = {
            "model": model,
            "messages": self._messages(system, messages),
            "stream": True,
            "options": self._options(max_tokens),
        }

        started = asyncio.get_event_loop().time()
        final: dict[str, Any] = {}
        async with self._http.stream("POST", "/api/chat", json=body) as response:
            response.raise_for_status()
            # Ollama streams newline-delimited JSON, one object per line.
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = (chunk.get("message") or {}).get("content", "")
                if token:
                    yield {"type": "token", "text": token}
                if chunk.get("done"):
                    final = chunk

        latency_ms = (asyncio.get_event_loop().time() - started) * 1000
        usage = self._usage_from(final, model, latency_ms)
        yield {
            "type": "usage",
            "usage": usage.as_dict(),
            "stop_reason": final.get("done_reason") or "end_turn",
        }

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
        """Schema-constrained generation via Ollama's `format` field.

        A small local model obeys a schema less reliably than a frontier one, so
        on a validation failure it gets exactly one retry with the error handed
        back to it. That is cheap here - the call costs nothing - and it turns
        most near-misses into valid objects.
        """
        model = model or self.settings.ollama_fast_model
        max_tokens = max_tokens or self.settings.fast_max_tokens
        json_schema = schema.model_json_schema()

        convo = self._messages(system, messages)
        last_error: Exception | None = None

        for attempt in range(2):
            body = {
                "model": model,
                "messages": convo,
                "stream": False,
                "format": json_schema,
                "options": self._options(max_tokens),
            }
            started = asyncio.get_event_loop().time()
            payload = await self._post("/api/chat", body)
            latency_ms = (asyncio.get_event_loop().time() - started) * 1000
            content = (payload.get("message") or {}).get("content", "") or "{}"

            try:
                return schema.model_validate_json(content), self._usage_from(
                    payload, model, latency_ms
                )
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("Ollama structured attempt %d failed: %s", attempt + 1, exc)
                if attempt == 0:
                    convo = convo + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                f"That did not validate against the schema: {exc}. "
                                "Return only a JSON object that matches it exactly."
                            ),
                        },
                    ]

        raise RuntimeError(f"Ollama could not produce valid {schema.__name__}: {last_error}")

    # -------------------------------------------------------- token count
    async def count_tokens(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
    ) -> int:
        """Approximate. Ollama has no count endpoint, and the tokeniser varies
        per model, so this is a ~4-chars-per-token estimate used only for
        budgeting - never for billing, which is zero anyway."""
        text = system + "".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        return max(1, len(text) // 4)


def _to_ollama_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Anthropic tool shape -> Ollama's OpenAI-compatible shape."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }
