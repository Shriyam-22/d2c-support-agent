"""LLM client.

The agent depends only on the small `LLM` protocol below, so the provider can
be swapped without touching the agent loop:
  - OpenAICompatLLM works with OpenAI and any OpenAI-compatible endpoint
    (set LLM_BASE_URL; several providers offer one, some with free tiers).
  - Tests use a scripted fake that returns pre-written responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, exactly as produced by the model


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    model: str

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OpenAICompatLLM:
    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 temperature: float = 0.2, timeout: float = 30.0):
        from openai import OpenAI  # imported lazily so tests run without it

        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout, max_retries=2)

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools or None, temperature=self.temperature)
        msg = resp.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in (msg.tool_calls or [])]
        usage = resp.usage
        return LLMResponse(
            content=msg.content,
            tool_calls=calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
