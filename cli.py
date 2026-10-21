"""Chat with the agent in the terminal.

    python cli.py            # normal chat
    python cli.py --debug    # also show tools called, flags, latency and cost
"""
import sys

from agent.core import build_agent, trim_history


def main() -> None:
    debug = "--debug" in sys.argv
    agent = build_agent()
    history: list[dict] = []
    print("Neem & Nest support (type 'exit' to quit). Try: 'mera order NN1002 kab aayega? phone 5678'\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        result = agent.respond(text, trim_history(history), session_id="cli")
        history = result.messages
        print(f"Neha: {result.reply}\n")
        if debug:
            t = result.trace
            print(f"  [lang={t.language} tools={t.tools_called} flags={result.flags} "
                  f"latency={t.total_latency_ms:.0f}ms tokens={t.input_tokens}+{t.output_tokens} "
                  f"cost=${t.cost_usd:.5f}]\n")


if __name__ == "__main__":
    main()
