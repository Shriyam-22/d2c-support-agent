"""Input and output guardrails.

Some rules are enforced in code because a prompt alone cannot guarantee them:
  - Order details are never returned unless the phone digits match
    (enforced inside the track_order tool, so the LLM never sees them).
  - Every price in a reply must appear in a tool result from the same turn.
    An LLM that "remembers" a price from training or makes one up is the
    most damaging hallucination in commerce support.
  - Order IDs mentioned alongside a delivery status must have been looked up
    and verified in the same turn.
  - Phone numbers and emails are redacted before anything is logged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 1000

PRICE_PATTERNS = [
    re.compile(r"(?:₹|\brs\.?|\binr)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:rupees|rupaye|rupay|/-)", re.IGNORECASE),
]
ORDER_ID = re.compile(r"\b#?(NN\d{3,})\b", re.IGNORECASE)
STATUS_WORDS = re.compile(
    r"\b(shipped|dispatched|delivered|out for delivery|in transit|cancelled|returned to origin|"
    r"pahunch|nikal gaya|deliver ho)\w*", re.IGNORECASE)
INJECTION = re.compile(
    r"(ignore (all |the )?(previous|above) instructions|system prompt|you are now|"
    r"developer mode|reveal your (prompt|instructions))", re.IGNORECASE)
PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _to_number(s: str) -> float:
    return float(s.replace(",", ""))


@dataclass
class InputCheck:
    text: str
    flags: list[str] = field(default_factory=list)
    reply: str | None = None  # if set, return this without calling the LLM


def check_input(text: str) -> InputCheck:
    text = (text or "").strip()
    if not text:
        return InputCheck(text, ["empty_input"], reply="Hi! How can I help you with your order or our products?")
    flags = []
    if len(text) > MAX_INPUT_CHARS:
        text, flags = text[:MAX_INPUT_CHARS], ["truncated_input"]
    if INJECTION.search(text):
        # We do not block (false positives annoy real customers); we flag it,
        # and the system prompt plus code-level guardrails limit the damage.
        flags.append("possible_prompt_injection")
    return InputCheck(text, flags)


@dataclass
class OutputCheck:
    ok: bool
    flags: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def prices_in(text: str) -> set[float]:
    found = set()
    for pat in PRICE_PATTERNS:
        for m in pat.finditer(text):
            try:
                found.add(_to_number(m.group(1)))
            except ValueError:
                pass
    return found


def check_output(reply: str, tool_results: list[dict], user_message: str = "") -> OutputCheck:
    import json

    flags, problems = [], []
    grounding_text = " ".join(json.dumps(r, ensure_ascii=False) for r in tool_results) + " " + user_message
    grounded_numbers = {_to_number(n) for n in NUMBER.findall(grounding_text)}

    ungrounded = sorted(p for p in prices_in(reply) if p not in grounded_numbers)
    if ungrounded:
        flags.append("ungrounded_price")
        problems.append(f"The reply states price(s) {ungrounded} that are not in any tool result.")

    verified_ids = {str(r.get("order_id", "")).upper() for r in tool_results if r.get("verified")}
    if STATUS_WORDS.search(reply):
        for oid in {m.upper() for m in ORDER_ID.findall(reply)}:
            if oid not in verified_ids:
                flags.append("ungrounded_order_status")
                problems.append(f"The reply gives a status for order {oid} without a verified track_order result.")

    return OutputCheck(ok=not problems, flags=flags, problems=problems)


def redact(text: str) -> str:
    return EMAIL.sub("[email]", PHONE.sub("[phone]", text or ""))
