from __future__ import annotations

import argparse

from core.llm import LLMInference


def test_aion_chat(device: str = "webgpu") -> None:
    """Run an end-to-end chat against the real local Aion model."""
    llm = LLMInference("aion", device=device)
    messages = [
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    reply = llm.chat(messages, max_new_tokens=32, temperature=0).strip()
    print(f"Aion ({device}): {reply}", flush=True)
    if "paris" not in reply.casefold():
        raise AssertionError(f"Expected a reply naming Paris, got {reply!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__ or "Run the local Aion chat test.")
    parser.add_argument("--device", choices=("webgpu", "cpu"), default="webgpu")
    test_aion_chat(parser.parse_args().device)
