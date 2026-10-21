"""Run the evaluation suite against the live agent.

    python -m evals.run_evals                  # all cases
    python -m evals.run_evals --case order_    # cases whose id starts with "order_"
    python -m evals.run_evals --repeat 3       # run each case 3 times (LLMs are not deterministic)

Writes a JSON report to evals/results/ and prints a summary table.
Each check is deterministic (tools called, substrings, language, guardrail
flags), so results are cheap to compute and easy to trust. Add an
LLM-as-judge score for tone/helpfulness once these pass consistently.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from agent.language import DEVANAGARI, detect_language

HERE = Path(__file__).resolve().parent


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    failures: list[str]
    reply: str
    tools_called: list[str]
    flags: list[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tool_check_passed: bool = True
    language_check_passed: bool | None = None


def load_cases(path: Path = HERE / "cases.yaml") -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def reply_language_ok(reply: str, expected: str) -> bool:
    if expected == "hi":
        return bool(DEVANAGARI.search(reply))
    if expected == "hinglish":
        return not DEVANAGARI.search(reply) and detect_language(reply) == "hinglish"
    return not DEVANAGARI.search(reply) and detect_language(reply) == "en"


def run_case(agent, case: dict) -> CaseResult:
    turns = case["input"] if isinstance(case["input"], list) else [case["input"]]
    history: list[dict] = []
    result = None
    latency = tokens_in = tokens_out = cost = 0.0
    for turn in turns:
        result = agent.respond(turn, history, session_id=f"eval:{case['id']}")
        history = result.messages
        t = result.trace
        latency += t.total_latency_ms
        tokens_in += t.input_tokens
        tokens_out += t.output_tokens
        cost += t.cost_usd

    reply, called, flags = result.reply, result.tools_called, result.flags
    low = reply.lower()
    failures: list[str] = []

    tool_ok = True
    for tool in case.get("expect_tools", []):
        if tool not in called:
            failures.append(f"expected tool {tool}, called {called}")
            tool_ok = False
    for tool in case.get("forbid_tools", []):
        if tool in called:
            failures.append(f"forbidden tool {tool} was called")
            tool_ok = False
    if case.get("expect_no_tools") and called:
        failures.append(f"expected no tools, called {called}")
        tool_ok = False

    options = case.get("must_include_any", [])
    if options and not any(o.lower() in low for o in options):
        failures.append(f"reply contains none of {options}")
    for bad in case.get("must_not_include", []):
        if bad.lower() in low:
            failures.append(f"reply contains forbidden text '{bad}'")

    lang_ok = None
    if "expect_language" in case:
        lang_ok = reply_language_ok(reply, case["expect_language"])
        if not lang_ok:
            failures.append(f"reply is not in {case['expect_language']}")

    for flag in case.get("expect_flags", []):
        if flag not in flags:
            failures.append(f"expected guardrail flag {flag}")
    if "fallback_used" in flags:
        failures.append("guardrail fallback used (reply failed grounding twice)")

    return CaseResult(case["id"], case["category"], not failures, failures, reply, called, flags,
                      round(latency, 1), int(tokens_in), int(tokens_out), round(cost, 6),
                      tool_ok, lang_ok)


def summarise(results: list[CaseResult]) -> dict:
    n = len(results) or 1
    latencies = sorted(r.latency_ms for r in results) or [0.0]
    lang = [r for r in results if r.language_check_passed is not None]
    by_cat: dict[str, list[CaseResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    return {
        "cases": len(results),
        "pass_rate": round(sum(r.passed for r in results) / n, 3),
        "tool_selection_accuracy": round(sum(r.tool_check_passed for r in results) / n, 3),
        "language_match_rate": round(sum(r.language_check_passed for r in lang) / len(lang), 3) if lang else None,
        "grounding_repairs": sum("repaired" in r.flags for r in results),
        "grounding_fallbacks": sum("fallback_used" in r.flags for r in results),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 1),
        "avg_tokens_per_case": round(sum(r.input_tokens + r.output_tokens for r in results) / n),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 5),
        "pass_rate_by_category": {c: round(sum(r.passed for r in rs) / len(rs), 3) for c, rs in sorted(by_cat.items())},
    }


def print_report(results: list[CaseResult], summary: dict) -> None:
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {r.id:<38} tools={r.tools_called} {r.latency_ms:>7.0f}ms")
        for f in r.failures:
            print(f"        - {f}")
        if not r.passed:
            print(f"        reply: {r.reply[:200]!r}")
    print("\nSummary")
    for k, v in summary.items():
        print(f"  {k:<26} {v}")


def main() -> None:
    from agent.core import build_agent

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="", help="only run cases whose id starts with this")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    agent = build_agent()
    cases = [c for c in load_cases() if c["id"].startswith(args.case)]
    results = [run_case(agent, c) for c in cases for _ in range(args.repeat)]
    summary = summarise(results)
    print_report(results, summary)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"model": getattr(agent.llm, "model", "?"), "summary": summary,
                               "results": [asdict(r) for r in results]}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nReport written to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
