"""Extract license information or driver names from text or transcribed speech."""

from __future__ import annotations

import argparse
import asyncio

from core.graph import AgentGraph
from core.llm import LLMInference


def run_chat(graph: AgentGraph) -> None:
    print("Interactive chat mode. Type 'exit' to quit.")

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        print("\nAssistant: ", end="", flush=True)
        result = asyncio.run(graph.run(prompt))
        print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?", help="Text or transcribed speech to process")
    parser.add_argument("--device", choices=("webgpu", "cpu"), default="webgpu")
    parser.add_argument("--config", help="Path to the graph tool configuration JSON")
    parser.add_argument(
        "--chat", action="store_true", help="Start an interactive chat"
    )
    args = parser.parse_args()

    graph = AgentGraph(LLMInference("aion", device=args.device), args.config)

    if args.chat or not args.message:
        run_chat(graph)
    else:
        print(asyncio.run(graph.run(args.message)))


if __name__ == "__main__":
    main()
