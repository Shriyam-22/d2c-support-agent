"""Per-turn tracing: latency, tokens, cost and every tool call.

Traces are appended to logs/traces.jsonl (PII redacted), one line per turn,
so cost and latency can be analysed with pandas or loaded into Langfuse/
LangSmith later. Prices are read from the environment because they differ by
provider and change often; set them from your provider's pricing page.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .guardrails import redact


def _price(name: str) -> float:
    try:
        return float(os.getenv(name, "0") or 0)
    except ValueError:
        return 0.0


@dataclass
class Span:
    kind: str  # "llm" | "tool"
    name: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    detail: dict = field(default_factory=dict)


@dataclass
class Trace:
    session_id: str
    user_message: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    language: str = "en"
    spans: list[Span] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    reply: str = ""
    started: float = field(default_factory=time.perf_counter)
    total_latency_ms: float = 0.0

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.spans)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.spans)

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * _price("PRICE_INPUT_PER_1M")
                + self.output_tokens * _price("PRICE_OUTPUT_PER_1M")) / 1_000_000

    @property
    def tools_called(self) -> list[str]:
        return [s.name for s in self.spans if s.kind == "tool"]

    def finish(self, reply: str) -> None:
        self.reply = reply
        self.total_latency_ms = (time.perf_counter() - self.started) * 1000

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("started")
        d["user_message"] = redact(self.user_message)
        d["reply"] = redact(self.reply)
        d.update(input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                 cost_usd=round(self.cost_usd, 6), tools_called=self.tools_called)
        return d

    def write(self, log_dir: Path | None) -> None:
        if log_dir is None:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "traces.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
