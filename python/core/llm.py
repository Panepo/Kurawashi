from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import NamedTuple

import onnxruntime_genai as og

from scripts.model_download import MODELS, MODELS_DIR, ModelEntry


class StreamMetrics(NamedTuple):
    first_token_s: float
    total_s: float
    tokens_per_s: float
    token_count: int


class LLMInference:
    """Run a local ONNX Runtime GenAI model with a fresh generator per request."""

    def __init__(self, model_name: str, device: str = "webgpu") -> None:
        entry = MODELS.get(model_name)
        if not isinstance(entry, ModelEntry):
            available = ", ".join(name for name, model in MODELS.items()
                                  if isinstance(model, ModelEntry))
            raise ValueError(f"Unknown LLM '{model_name}'. Available: {available}")
        device = device.lower()
        if device not in {"webgpu", "cpu"}:
            raise ValueError(f"Unsupported device '{device}'; use webgpu or cpu.")

        model_path = MODELS_DIR / entry.folder
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        self.model_name = model_name
        self.device = device
        settings = json.loads((model_path / "genai_config.json").read_text(encoding="utf-8"))
        self._context_length: int = settings["model"]["context_length"]
        self._chat_template = (model_path / "chat_template.jinja").read_text(encoding="utf-8")
        config = og.Config(str(model_path))
        config.clear_providers()
        if device == "webgpu":
            config.append_provider("webgpu")
        self._model = og.Model(config)
        self._tokenizer = og.Tokenizer(self._model)

    def _generate_tokens(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int
    ) -> Iterator[int]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if not temperature >= 0 or not 0 < top_p <= 1 or top_k < 0:
            raise ValueError("Require temperature >= 0, 0 < top_p <= 1, and top_k >= 0.")
        input_ids = self._tokenizer.encode(prompt)
        if len(input_ids) == 0:
            raise ValueError("The prompt must encode to at least one token.")
        max_length = len(input_ids) + max_new_tokens
        if max_length > self._context_length:
            raise ValueError(f"Prompt and response budget exceed {self._context_length} tokens.")

        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=max_length, do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p, top_k=top_k,
        )
        generator = og.Generator(self._model, params)
        generator.append_tokens(input_ids)
        while not generator.is_done():
            generator.generate_next_token()
            yield int(generator.get_next_tokens()[0])

    def _chat_prompt(self, messages: list[dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(
            json.dumps(messages), template_str=self._chat_template, add_generation_prompt=True
        )

    def chat_with_tools(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]],
        max_new_tokens: int = 512, temperature: float = 0,
    ) -> str:
        """Render structured messages and tool definitions using the model's template."""
        prompt = self._tokenizer.apply_chat_template(
            json.dumps(messages), template_str=self._chat_template,
            tools=json.dumps(tools), add_generation_prompt=True,
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def generate(
        self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 50,
    ) -> str:
        """Return only the tokens generated after the prompt."""
        tokens = list(self._generate_tokens(prompt, max_new_tokens, temperature, top_p, top_k))
        return self._tokenizer.decode(tokens)

    def generate_stream(
        self, prompt: str, callback: Callable[[str], bool] | None = None,
        max_new_tokens: int = 512, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 50,
    ) -> StreamMetrics:
        """Stream text to a callback (True stops generation), or print it."""
        start = time.perf_counter()
        first_token_s = None
        count = 0
        stream = self._tokenizer.create_stream()
        for token in self._generate_tokens(prompt, max_new_tokens, temperature, top_p, top_k):
            if first_token_s is None:
                first_token_s = time.perf_counter() - start
            count += 1
            text = stream.decode(token)
            if not text:
                continue
            if callback is None:
                print(text, end="", flush=True)
            elif callback(text):
                break
        elapsed = time.perf_counter() - start
        return StreamMetrics(
            elapsed if first_token_s is None else first_token_s,
            elapsed,
            count / elapsed if elapsed else 0.0,
            count,
        )

    def chat(
        self, messages: list[dict[str, str]], max_new_tokens: int = 512,
        temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
    ) -> str:
        """Apply the model's chat template and return the assistant reply."""
        return self.generate(self._chat_prompt(messages), max_new_tokens, temperature, top_p, top_k)

    def chat_stream(
        self, messages: list[dict[str, str]], callback: Callable[[str], bool] | None = None,
        max_new_tokens: int = 512, temperature: float = 0.7,
        top_p: float = 0.9, top_k: int = 50,
    ) -> StreamMetrics:
        """Apply the model's chat template and stream the assistant reply."""
        return self.generate_stream(
            self._chat_prompt(messages), callback, max_new_tokens, temperature, top_p, top_k
        )
