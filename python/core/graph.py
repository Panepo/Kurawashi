from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from agent_framework import (
    BaseChatClient,
    ChatResponse,
    Content,
    Executor,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)


class ToolBackend(Protocol):
    def chat_with_tools(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]],
        max_new_tokens: int = 512, temperature: float = 0,
    ) -> str: ...


_INSTRUCTIONS = (
    "Extract license information from the user's message using the available tools. "
    "Choose query_license_plate for a plate or query_driver_license for a driver license. "
    "Pass number_text as the exact words or characters from the user's message, "
    "including every letter and digit, not a rewritten or inferred number. "
    "For example, for 'plate alfa bravo charlie 123 in California', number_text is "
    "'alfa bravo charlie 123' and state is 'California'. "
    "Pass state only if explicitly stated. "
    "If the information is missing or ambiguous, ask for clarification. "
    "Do not invent information."
)
_TOOL_DESCRIPTIONS = {
    "query_license_plate": "Extract a license plate and optional state from the user message.",
    "query_driver_license": "Extract a driver license number from the user message.",
}
_DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_CALL_PATTERN = re.compile(r"<\|tool_call\|>(.*?)<\|/tool_call\|>", re.DOTALL)


def _tool_schema(name: str) -> dict[str, object]:
    properties: dict[str, object] = {
        "number_text": {
            "type": "string",
            "description": "Exact words or characters for the number from the user's message.",
        }
    }
    if name == "query_license_plate":
        properties["state"] = {
            "type": ["string", "null"],
            "description": "Exact state mentioned by the user, or null if not mentioned.",
        }
    return {
        "name": name,
        "description": _TOOL_DESCRIPTIONS[name],
        "parameters": {
            "type": "object", "properties": properties,
            "required": ["number_text"], "additionalProperties": False,
        },
    }


def _aion_messages(messages: Sequence[Message]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for message in messages:
        item: dict[str, object] = {"role": message.role, "content": message.text}
        calls: list[dict[str, object]] = []
        for content in message.contents:
            if content.type == "function_call":
                calls.append({"name": content.name, "arguments": content.arguments})
            elif content.type == "function_result":
                item["content"] = (
                    content.result if isinstance(content.result, str)
                    else json.dumps(content.result, ensure_ascii=False)
                )
            elif content.type != "text":
                raise ValueError(f"Unsupported message content: {content.type}")
        if calls:
            item["tool_calls"] = calls
        result.append(item)
    return result


class LocalChatClient(BaseChatClient):
    """Translate Agent Framework messages to the local Aion tool protocol."""

    def __init__(self, backend: ToolBackend, tools: list[dict[str, object]]) -> None:
        super().__init__()
        self._backend = backend
        self._tools = tools

    async def _inner_get_response(
        self, *, messages: Sequence[Message], stream: bool,
        options: Mapping[str, Any], **kwargs: Any,
    ) -> ChatResponse:
        if stream:
            raise NotImplementedError("Streaming tool calls are not supported.")
        await self._validate_options(options)
        reply = await asyncio.to_thread(
            self._backend.chat_with_tools, _aion_messages(messages), self._tools
        )
        matches = list(_CALL_PATTERN.finditer(reply))
        remaining = _CALL_PATTERN.sub("", reply).strip()
        if "<|tool_call|>" in remaining or "<|/tool_call|>" in remaining:
            raise ValueError("Malformed tool-call response from model.")
        contents: list[Content] = []
        if remaining:
            contents.append(Content.from_text(remaining))
        for match in matches:
            try:
                call = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                raise ValueError("Model returned invalid tool-call JSON.") from exc
            if (
                not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str)
                or not isinstance(call["arguments"], dict)
            ):
                raise ValueError("Model returned an invalid tool call.")
            contents.append(Content.from_function_call(
                str(uuid.uuid4()), call["name"], arguments=call["arguments"]
            ))
        if not contents:
            raise ValueError("Model returned an empty response.")
        return ChatResponse(messages=[Message("assistant", contents)])


def _load_alphabet(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("NATO alphabet must be a non-empty JSON object.")
    alphabet: dict[str, str] = {}
    for word, letter in data.items():
        if (
            not isinstance(word, str) or not re.fullmatch(r"[a-z]+(?:-[a-z]+)*", word)
            or not isinstance(letter, str) or not re.fullmatch(r"[A-Z]", letter)
        ):
            raise ValueError(f"Invalid NATO alphabet entry: {word!r}.")
        alphabet[word] = letter
    return alphabet


def _normalise_number(value: str, alphabet: Mapping[str, str]) -> str:
    if not value.strip() or not re.fullmatch(r"[A-Za-z0-9 -]+", value):
        raise ValueError("Number must contain only letters, digits, spaces, or hyphens.")
    parts = value.split()
    result: list[str] = []
    for part in parts:
        word = part.casefold()
        if word in alphabet:
            result.append(alphabet[word])
        elif word in _DIGITS:
            result.append(_DIGITS[word])
        elif re.fullmatch(r"[0-9]+", part):
            result.append(part)
        elif re.fullmatch(r"[A-Za-z0-9]+", part) and (
            len(part) == 1 or any(char.isdigit() for char in part) or part.isupper()
        ):
            result.append(part.upper())
        else:
            raise ValueError(f"Ambiguous number component: {part!r}.")
    return "".join(result)


def _assert_in_source(value: str, source: str, field_name: str) -> None:
    if not value.strip() or not re.search(
        r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)", source, re.IGNORECASE
    ):
        raise ValueError(f"{field_name} must be taken verbatim from the user's message.")


def _assert_complete_number(value: str, source: str, alphabet: Mapping[str, str]) -> None:
    pattern = r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)"
    matches = list(re.finditer(pattern, source, re.IGNORECASE))
    if len(matches) != 1:
        raise ValueError("Number occurs more than once; clarification is required.")
    match = matches[0]
    before = re.search(r"([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)\s+$", source[:match.start()])
    after = re.match(r"\s+([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)", source[match.end():])
    for adjacent in (before, after):
        if adjacent:
            part = adjacent.group(1)
            if (
                part.casefold() in alphabet or part.casefold() in _DIGITS
                or part.isdigit() or (len(part) == 1 and part.isalnum())
                or (part.isalnum() and any(char.isdigit() for char in part))
            ):
                raise ValueError("Number may be incomplete; clarification is required.")


@dataclass(frozen=True)
class _RunState:
    source: str
    messages: list[Message]
    call: Content | None = None
    results: list[dict[str, str | None]] = field(default_factory=list)
    answer: str = ""


class _AgentExecutor(Executor):
    def __init__(self, client: LocalChatClient, max_tool_calls: int) -> None:
        super().__init__(id="agent")
        self._client = client
        self._max_tool_calls = max_tool_calls

    @handler
    async def handle(self, state: _RunState, ctx: WorkflowContext[_RunState]) -> None:
        response = await self._client.get_response(state.messages)
        if len(response.messages) != 1:
            raise ValueError("Expected exactly one assistant message.")
        message = response.messages[0]
        calls = [content for content in message.contents if content.type == "function_call"]
        if len(calls) > 1:
            raise ValueError("Only one tool call per model response is supported.")
        if calls and len(state.results) >= self._max_tool_calls:
            raise ValueError("Maximum tool-call count exceeded.")
        await ctx.send_message(replace(
            state, messages=[*state.messages, message],
            call=calls[0] if calls else None, answer=message.text,
        ))


class _ToolExecutor(Executor):
    def __init__(self, tools: set[str], alphabet: Mapping[str, str]) -> None:
        super().__init__(id="tool")
        self._tools = tools
        self._alphabet = alphabet

    @handler
    async def handle(self, state: _RunState, ctx: WorkflowContext[_RunState]) -> None:
        call = state.call
        if call is None or call.name not in self._tools or not call.call_id:
            raise ValueError("Model requested an unknown or disabled tool.")
        args = call.arguments
        allowed = {"number_text", "state"} if call.name == "query_license_plate" else {"number_text"}
        if not isinstance(args, Mapping) or set(args) - allowed:
            raise ValueError("Invalid tool arguments.")
        number = args.get("number_text")
        if not isinstance(number, str):
            raise ValueError("Tool requires a number_text string.")
        _assert_in_source(number, state.source, "number_text")
        _assert_complete_number(number, state.source, self._alphabet)
        normalised = _normalise_number(number, self._alphabet)
        if call.name == "query_license_plate":
            region = args.get("state")
            if region is not None:
                if not isinstance(region, str):
                    raise ValueError("State must be a string or null.")
                region = region.strip()
                if not re.fullmatch(r"[^\W\d_]+(?:[ -][^\W\d_]+)*", region):
                    raise ValueError("State must contain letters, spaces, or hyphens.")
                _assert_in_source(region, state.source, "state")
            result = {"license_plate_number": normalised, "state": region}
        else:
            result = {"driver_license_number": normalised}
        tool_message = Message("tool", [Content.from_function_result(
            call.call_id, result=result
        )])
        await ctx.send_message(replace(
            state, results=[*state.results, result],
            messages=[*state.messages, tool_message], call=None,
        ))


class _OutputExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="output")

    @handler
    async def handle(self, state: _RunState, ctx: WorkflowContext[None, str]) -> None:
        if state.results:
            result = state.results[0] if len(state.results) == 1 else state.results
            output = json.dumps(result, ensure_ascii=False)
        elif state.answer:
            output = state.answer
        else:
            raise ValueError("Agent returned no answer.")
        await ctx.yield_output(output)


class AgentGraph:
    """Configured local license-extraction workflow."""

    def __init__(self, backend: ToolBackend, config_path: str | Path | None = None) -> None:
        self._backend = backend
        self.load_config(config_path or Path(__file__).with_name("tools.json"))

    def load_config(self, path: str | Path) -> None:
        config_path = Path(path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or set(config) != {
            "tools", "alphabet_file", "max_tool_calls"
        }:
            raise ValueError("Config must contain tools, alphabet_file, and max_tool_calls.")
        tools = config["tools"]
        if (
            not isinstance(tools, list) or not tools
            or any(not isinstance(name, str) or name not in _TOOL_DESCRIPTIONS for name in tools)
            or len(set(tools)) != len(tools)
        ):
            raise ValueError("Config must list unique, known tools.")
        alphabet_file = config["alphabet_file"]
        if not isinstance(alphabet_file, str) or not alphabet_file:
            raise ValueError("alphabet_file must be a non-empty path.")
        max_calls = config["max_tool_calls"]
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError("max_tool_calls must be a positive integer.")
        alphabet = _load_alphabet(config_path.parent / alphabet_file)
        client = LocalChatClient(self._backend, [_tool_schema(name) for name in tools])
        agent = _AgentExecutor(client, max_calls)
        tool = _ToolExecutor(set(tools), alphabet)
        output = _OutputExecutor()
        builder = WorkflowBuilder(start_executor=agent, max_iterations=2 * max_calls + 3)
        builder.add_edge(agent, tool, condition=lambda state: state.call is not None)
        builder.add_edge(agent, output, condition=lambda state: state.call is None)
        builder.add_edge(tool, agent)
        self._workflow = builder.build()

    async def run(self, message: str) -> str:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("User message must be a non-empty string.")
        state = _RunState(
            source=message,
            messages=[Message("system", [_INSTRUCTIONS]), Message("user", [message])],
        )
        result = await self._workflow.run(state)
        outputs = result.get_outputs()
        if len(outputs) != 1 or not isinstance(outputs[0], str):
            raise RuntimeError("Workflow did not produce exactly one output.")
        return outputs[0]
