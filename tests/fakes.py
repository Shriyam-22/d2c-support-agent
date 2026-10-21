"""A scripted LLM for tests: returns pre-written responses in order and
records every request, so the agent loop can be tested without an API key."""
from __future__ import annotations

import json
from pathlib import Path

from agent.core import SupportAgent
from agent.llm import LLMResponse, ToolCall
from agent.retrieval import BM25Retriever, load_chunks
from agent.store import MockShopifyStore
from agent.tools import build_registry

DATA = Path(__file__).resolve().parent.parent / "data"


class ScriptedLLM:
    model = "scripted"

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.requests: list[list[dict]] = []

    def complete(self, messages, tools):
        self.requests.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)


def call(name: str, args: dict, id: str = "call_1") -> LLMResponse:
    return LLMResponse(content=None, tool_calls=[ToolCall(id, name, json.dumps(args))],
                       input_tokens=100, output_tokens=20)


def say(text: str) -> LLMResponse:
    return LLMResponse(content=text, input_tokens=150, output_tokens=40)


def make_registry():
    return build_registry(MockShopifyStore(DATA / "orders.json"), BM25Retriever(load_chunks(DATA)))


def make_agent(responses: list[LLMResponse], max_steps: int = 5) -> tuple[SupportAgent, ScriptedLLM]:
    llm = ScriptedLLM(responses)
    return SupportAgent(llm, make_registry(), max_steps=max_steps, log_dir=None), llm
