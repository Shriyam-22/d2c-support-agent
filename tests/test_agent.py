"""Agent-loop tests with a scripted LLM (no API key needed)."""
import unittest

from agent.core import trim_history
from agent.prompts import FALLBACK_REPLY
from app.whatsapp import parse_webhook
from evals.run_evals import load_cases, run_case, summarise
from tests.fakes import call, make_agent, say


class AgentLoopTests(unittest.TestCase):
    def test_tool_call_then_answer(self):
        agent, llm = make_agent([
            call("track_order", {"order_id": "NN1002", "phone_last4": "5678"}),
            say("Your order NN1002 is in transit with Blue Dart and should arrive by 23 Oct."),
        ])
        result = agent.respond("Where is NN1002? phone 5678")
        self.assertIn("Blue Dart", result.reply)
        self.assertEqual(result.tools_called, ["track_order"])
        self.assertEqual(result.flags, [])
        # The tool result was sent back to the model in the second request.
        tool_msgs = [m for m in llm.requests[1] if m["role"] == "tool"]
        self.assertIn('"carrier": "Blue Dart"', tool_msgs[0]["content"])

    def test_system_prompt_sets_reply_language(self):
        agent, llm = make_agent([say("Aapka order jaldi aayega.")])
        agent.respond("mera order kab aayega?")
        self.assertIn("Hinglish", llm.requests[0][0]["content"])

    def test_hallucinated_price_is_repaired(self):
        agent, _ = make_agent([
            call("search_products", {"query": "vitamin c serum"}),
            say("The Vitamin C serum is ₹599."),          # wrong price
            say("The Vitamin C + Amla Serum is ₹799."),   # repaired
        ])
        result = agent.respond("price of vitamin c serum?")
        self.assertIn("₹799", result.reply)
        self.assertIn("ungrounded_price", result.flags)
        self.assertIn("repaired", result.flags)

    def test_failed_repair_uses_safe_fallback(self):
        agent, _ = make_agent([say("It costs ₹123."), say("Still ₹123.")])
        result = agent.respond("how much is the face wash?")
        self.assertEqual(result.reply, FALLBACK_REPLY["en"])
        self.assertIn("fallback_used", result.flags)

    def test_max_steps_stops_tool_loops(self):
        loop = [call("search_policies", {"query": "returns"}, id=f"c{i}") for i in range(3)]
        agent, _ = make_agent(loop, max_steps=3)
        result = agent.respond("returns?")
        self.assertIn("max_steps_reached", result.flags)

    def test_history_carries_into_next_turn(self):
        agent, llm = make_agent([
            say("Sure! Please share the last 4 digits of your phone number."),
            call("track_order", {"order_id": "NN1003", "phone_last4": "6655"}),
            say("Order NN1003 is being packed and should arrive by 26 Oct."),
        ])
        first = agent.respond("Where is my order NN1003?")
        second = agent.respond("6655", first.messages)
        self.assertIn("packed", second.reply)
        self.assertIn("NN1003", llm.requests[1][1]["content"])  # earlier user turn was sent

    def test_trim_history_starts_at_user_message(self):
        msgs = [{"role": "user", "content": "1"}, {"role": "assistant", "content": "a"},
                {"role": "tool", "content": "t"}, {"role": "user", "content": "2"},
                {"role": "assistant", "content": "b"}]
        trimmed = trim_history(msgs, max_messages=3)
        self.assertEqual(trimmed[0], {"role": "user", "content": "2"})

    def test_usage_and_tools_are_traced(self):
        agent, _ = make_agent([call("search_policies", {"query": "cod"}), say("COD is available up to ₹2,000.")])
        result = agent.respond("is COD available?")
        t = result.trace
        self.assertEqual(t.tools_called, ["search_policies"])
        self.assertEqual(t.input_tokens, 250)
        self.assertGreater(t.total_latency_ms, 0)


class WhatsAppParsingTests(unittest.TestCase):
    def test_parses_text_messages_and_ignores_others(self):
        payload = {"entry": [{"changes": [{"value": {
            "contacts": [{"wa_id": "919876501234", "profile": {"name": "Priya"}}],
            "messages": [
                {"from": "919876501234", "id": "wamid.1", "type": "text", "text": {"body": "order kab aayega"}},
                {"from": "919876501234", "id": "wamid.2", "type": "image", "image": {}},
            ]}}]}]}
        msgs = parse_webhook(payload)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].text, "order kab aayega")
        self.assertEqual(msgs[0].name, "Priya")

    def test_status_updates_produce_no_messages(self):
        payload = {"entry": [{"changes": [{"value": {"statuses": [{"status": "delivered"}]}}]}]}
        self.assertEqual(parse_webhook(payload), [])


class EvalRunnerTests(unittest.TestCase):
    def test_cases_file_is_valid(self):
        cases = load_cases()
        ids = [c["id"] for c in cases]
        self.assertEqual(len(ids), len(set(ids)), "case ids must be unique")
        for c in cases:
            self.assertIn("input", c)
            self.assertIn("category", c)

    def test_run_case_pass_and_fail(self):
        case = {"id": "t", "category": "orders", "input": "Where is NN1002? phone 5678",
                "expect_tools": ["track_order"], "must_include_any": ["Blue Dart"], "expect_language": "en"}
        agent, _ = make_agent([call("track_order", {"order_id": "NN1002", "phone_last4": "5678"}),
                               say("Order NN1002 is in transit with Blue Dart.")])
        good = run_case(agent, case)
        self.assertTrue(good.passed, good.failures)

        agent, _ = make_agent([say("Please share your order details.")])
        bad = run_case(agent, case)
        self.assertFalse(bad.passed)
        self.assertFalse(bad.tool_check_passed)

        summary = summarise([good, bad])
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["pass_rate_by_category"], {"orders": 0.5})


if __name__ == "__main__":
    unittest.main()
