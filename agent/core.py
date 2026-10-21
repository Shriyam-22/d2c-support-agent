"""The agent loop.

One customer message goes through:
  1. input guardrails  (empty, too long, prompt-injection flag)
  2. language detection (English / Hindi / Hinglish) -> reply instruction
  3. tool-calling loop  (LLM -> tools -> LLM ..., capped at max_steps)
  4. output guardrails  (price and order-status grounding)
     -> one repair attempt if the check fails -> safe fallback if it fails again
  5. trace written      (latency, tokens, cost, tools, flags; PII redacted)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import guardrails
from .config import Settings
from .language import LANGUAGE_INSTRUCTIONS, detect_language
from .llm import LLM, LLMResponse, OpenAICompatLLM
from .prompts import FALLBACK_REPLY, REPAIR_PROMPT, SYSTEM_PROMPT
from .retrieval import BM25Retriever, EmbeddingRetriever, Retriever, load_chunks, openai_embed_fn
from .store import MockShopifyStore, OrderStore
from .tools import ToolRegistry, build_registry
from .tracing import Span, Trace


@dataclass
class AgentResult:
    reply: str
    trace: Trace
    flags: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # full history to carry into the next turn

    @property
    def tools_called(self) -> list[str]:
        return self.trace.tools_called


class SupportAgent:
    def __init__(self, llm: LLM, registry: ToolRegistry, max_steps: int = 5, log_dir=None):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.log_dir = log_dir

    # ------------------------------------------------------------ public

    def respond(self, user_message: str, history: list[dict] | None = None,
                session_id: str = "local") -> AgentResult:
        check = guardrails.check_input(user_message)
        language = detect_language(check.text)
        trace = Trace(session_id=session_id, user_message=check.text, language=language, flags=list(check.flags))

        if check.reply is not None:
            return self._finish(trace, check.reply, history or [])

        system = SYSTEM_PROMPT.format(language_instruction=LANGUAGE_INSTRUCTIONS[language])
        prior = [m for m in (history or []) if m.get("role") != "system"]
        messages = [{"role": "system", "content": system}, *prior, {"role": "user", "content": check.text}]
        tool_results: list[dict] = []

        reply = self._run_loop(messages, tool_results, trace)
        if reply is None:
            trace.flags.append("max_steps_reached")
            reply = FALLBACK_REPLY[language]
        else:
            reply = self._check_and_repair(reply, messages, tool_results, check.text, trace, language)

        messages.append({"role": "assistant", "content": reply})
        return self._finish(trace, reply, messages[1:])

    # ------------------------------------------------------------ internals

    def _call_llm(self, messages: list[dict], trace: Trace) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self.llm.complete(messages, self.registry.schemas())
        trace.spans.append(Span("llm", getattr(self.llm, "model", "llm"), (time.perf_counter() - t0) * 1000,
                                resp.input_tokens, resp.output_tokens,
                                {"tool_calls": [c.name for c in resp.tool_calls]}))
        return resp

    def _run_loop(self, messages: list[dict], tool_results: list[dict], trace: Trace) -> str | None:
        for _ in range(self.max_steps):
            resp = self._call_llm(messages, trace)
            if not resp.tool_calls:
                return (resp.content or "").strip()
            messages.append({
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments}} for c in resp.tool_calls],
            })
            for call in resp.tool_calls:
                t0 = time.perf_counter()
                result = self.registry.execute(call.name, call.arguments)
                trace.spans.append(Span("tool", call.name, (time.perf_counter() - t0) * 1000,
                                        detail={"arguments": guardrails.redact(call.arguments),
                                                "error": result.get("error")}))
                tool_results.append(result)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
        return None

    def _check_and_repair(self, reply: str, messages: list[dict], tool_results: list[dict],
                          user_text: str, trace: Trace, language: str) -> str:
        out = guardrails.check_output(reply, tool_results, user_text)
        if out.ok:
            return reply
        trace.flags.extend(out.flags)
        repair_messages = [*messages, {"role": "assistant", "content": reply},
                           {"role": "user", "content": REPAIR_PROMPT.format(problems="\n".join(out.problems))}]
        resp = self._call_llm(repair_messages, trace)
        # A repair that tries to call tools again is treated as a failed repair.
        repaired = "" if resp.tool_calls else (resp.content or "").strip()
        if repaired and guardrails.check_output(repaired, tool_results, user_text).ok:
            trace.flags.append("repaired")
            return repaired
        trace.flags.append("fallback_used")
        return FALLBACK_REPLY[language]

    def _finish(self, trace: Trace, reply: str, messages: list[dict]) -> AgentResult:
        trace.finish(reply)
        trace.write(self.log_dir)
        return AgentResult(reply=reply, trace=trace, flags=trace.flags, messages=messages)


def trim_history(messages: list[dict], max_messages: int = 20) -> list[dict]:
    """Keep the most recent messages, always starting at a user message so an
    assistant tool call is never separated from its tool results."""
    if len(messages) <= max_messages:
        return messages
    tail = messages[-max_messages:]
    for i, m in enumerate(tail):
        if m.get("role") == "user":
            return tail[i:]
    return []


def build_agent(settings: Settings | None = None, llm: LLM | None = None,
                store: OrderStore | None = None) -> SupportAgent:
    """Wire up the default agent from environment settings."""
    s = settings or Settings.from_env()
    if llm is None:
        if not s.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and add your key.")
        llm = OpenAICompatLLM(s.llm_model, s.llm_api_key, s.llm_base_url)
    chunks = load_chunks(s.data_dir)
    retriever: Retriever
    if s.retriever == "embedding":
        retriever = EmbeddingRetriever(chunks, openai_embed_fn(llm.client, s.embedding_model))  # type: ignore[attr-defined]
    else:
        retriever = BM25Retriever(chunks)
    store = store or MockShopifyStore(s.data_dir / "orders.json")
    return SupportAgent(llm, build_registry(store, retriever), max_steps=s.max_steps, log_dir=s.log_dir)
