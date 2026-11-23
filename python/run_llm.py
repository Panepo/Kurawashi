"""Run a local ONNX Runtime GenAI language model."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from scripts.model_download import MODELS, ModelEntry

if TYPE_CHECKING:
    from core.llm import LLMInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=[name for name, entry in MODELS.items() if isinstance(entry, ModelEntry)],
        default="aion",
        help="Text generation model alias",
    )
    parser.add_argument(
        "--device", type=str.lower, choices=("webgpu", "cpu"), default="webgpu",
        help="Execution device",
    )
    parser.add_argument(
        "--prompt", default="Explain what ONNX Runtime is in two sentences.",
        help="One-shot prompt",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, dest="max_new_tokens",
        help="Maximum number of new tokens",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--stream", action="store_true", help="Stream the one-shot response")
    parser.add_argument(
        "--chat", action="store_true", help="Start an interactive chat that always streams"
    )
    return parser.parse_args()


def run_chat(llm: LLMInference, max_new_tokens: int, temperature: float) -> None:
    print("Interactive chat mode. Type 'exit' to quit.")
    history: list[dict[str, str]] = []

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        history.append({"role": "user", "content": prompt})
        parts: list[str] = []

        def display(text: str) -> bool:
            parts.append(text)
            print(text, end="", flush=True)
            return False

        print("Assistant: ", end="", flush=True)
        metrics = llm.chat_stream(
            history, callback=display, max_new_tokens=max_new_tokens, temperature=temperature
        )
        history.append({"role": "assistant", "content": "".join(parts)})
        print(
            f"\n[First token: {metrics.first_token_s:.3f}s | "
            f"Total: {metrics.total_s:.3f}s | {metrics.tokens_per_s:.1f} tok/s]\n"
        )


def main() -> None:
    args = parse_args()

    from core.llm import LLMInference

    print(f"Loading model '{args.model}' on {args.device} ...", flush=True)
    llm = LLMInference(args.model, device=args.device)
    if args.chat:
        run_chat(llm, args.max_new_tokens, args.temperature)
        return

    print(f"Prompt: {args.prompt}\n\n--- Response ---")
    if args.stream:
        llm.generate_stream(
            args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )
        print()
    else:
        print(llm.generate(
            args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature
        ))


if __name__ == "__main__":
    main()
