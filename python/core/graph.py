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

from tools.query_driver_by_name import query_driver_by_name
from tools.query_driver_license import query_driver_license
from tools.query_license_plate import query_license_plate
from tools.regions import resolve_us_region, us_region_is_mentioned


class ToolBackend(Protocol):
    def chat_with_tools(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]],
        max_new_tokens: int = 512, temperature: float = 0,
    ) -> str: ...


_INSTRUCTIONS = (
    "You are a helpful assistant. Use the available tools only when the user asks to "
    "extract a license plate, driver license number, or query a driver by name. For messages unrelated to those "
    "tools, respond naturally and conversationally without calling a tool. "
    "Choose query_license_plate for a plate or query_driver_license for a driver license. "
    "Choose query_driver_by_name for a driver name. Extract first_name and last_name "
    "verbatim from the user's message, preserving spelling, punctuation, and compound names. "
    "For 'query driver John Smith', first_name is 'John' and last_name is 'Smith'. "
    "If either name is missing or the first/last name split is ambiguous, ask for clarification. "
    "Call query_driver_by_name separately for each driver requested. "
    "Call the appropriate tool separately for each number requested, including multiple "
    "numbers of the same type. "
    "Pass number_text as the exact words or characters from the user's message, "
    "including every letter and digit, not a rewritten or inferred number. "
    "For example, for 'plate alfa bravo charlie 123 in California', number_text is "
    "'alfa bravo charlie 123' and state is 'California'. For 'driver license D123 in "
    "California', number_text is 'D123' and state is 'California'. "
    "Recognize standard two-letter U.S. state abbreviations as states, even when they "
    "appear before the license type. For 'query WA license plate aaadsdas3231', pass "
    "number_text 'aaadsdas3231' and state 'WA'. Convert state names to USPS "
    "abbreviations. For 'query Seattle license plate ABC123', pass state 'Seattle'; "
    "the application resolves it to 'WA'. For other cities, pass the city name only "
    "when it is supported and unambiguous; otherwise ask for the state. "
    "For '2 plates DEF456 and alice boy david xray 222', call query_license_plate twice: "
    "once with number_text 'DEF456' and once with number_text 'alice boy david xray 222'. "
    "Pass state when a state or supported city is explicitly mentioned. "
    "If the information is missing or ambiguous, ask for clarification. "
    "Do not invent information."
)
_TOOL_DESCRIPTIONS = {
    "query_driver_by_name": "Extract a driver's first name and last name from the user message.",
    "query_license_plate": "Extract a license plate and optional state from the user message.",
    "query_driver_license": "Extract a driver license number and optional state from the user message.",
}
_DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_CALL_PATTERN = re.compile(r"<\|tool_call\|>(.*?)<\|/tool_call\|>", re.DOTALL)


def _tool_schema(name: str) -> dict[str, object]:
    if name == "query_driver_by_name":
        return {
            "name": name,
            "description": _TOOL_DESCRIPTIONS[name],
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {
                        "type": "string",
                        "description": "Exact first name from the user's message.",
                    },
                    "last_name": {
                        "type": "string",
                        "description": "Exact last name from the user's message.",
                    },
                },
                "required": ["first_name", "last_name"],
                "additionalProperties": False,
            },
        }
    properties: dict[str, object] = {
        "number_text": {
            "type": "string",
            "description": "Exact words or characters for the number from the user's message.",
        }
    }
    properties["state"] = {
        "type": ["string", "null"],
        "description": (
            "Exact state name, standard two-letter U.S. state abbreviation, or city "
            "mentioned by the user, or null if no location is mentioned."
        ),
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


def _region_from_request(source: str) -> str | None:
    leading = re.search(
        r"\b(?:query|search|find|lookup)\s+(.+?)\s+"
        r"(?:license\s+plate|plate|driver\s+license)\b",
        source,
        re.IGNORECASE,
    )
    if leading:
        candidate = re.sub(r"^(?:(?:in|for|the|a|an)\s+)+", "", leading.group(1).strip(), flags=re.IGNORECASE)
        if candidate.casefold() not in {"driver", "license"} and not re.search(
            r"\b(?:driver|license)\b", candidate, re.IGNORECASE
        ):
            return candidate.strip(" ,.;:!?")
    trailing = re.search(
        r"\b(?:in|from)\s+([A-Za-z][A-Za-z .'-]*?)\s*[,.!?]*$",
        source,
        re.IGNORECASE,
    )
    return trailing.group(1).strip() if trailing else None


def _extract_driver_name(args: Mapping[str, object], source: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for field_name in ("first_name", "last_name"):
        value = args.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} is missing.")
        value = value.strip()
        if not re.fullmatch(r"[^\W\d_]+(?:[ '\u2019-][^\W\d_]+)*", value):
            raise ValueError(f"{field_name} must contain letters, spaces, apostrophes, or hyphens.")
        _assert_in_source(value, source, field_name)
        names[field_name] = value
    return query_driver_by_name(names["first_name"], names["last_name"])


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
    calls: list[Content] = field(default_factory=list)
    results: list[dict[str, object]] = field(default_factory=list)
    answer: str = ""
    clarification: str = ""


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
        if len(state.results) + len(calls) > self._max_tool_calls:
            raise ValueError("Maximum tool-call count exceeded.")
        await ctx.send_message(replace(
            state, messages=[*state.messages, message],
            calls=calls, answer=message.text,
        ))


class _ToolExecutor(Executor):
    def __init__(self, tools: set[str], alphabet: Mapping[str, str]) -> None:
        super().__init__(id="tool")
        self._tools = tools
        self._alphabet = alphabet

    @handler
    async def handle(self, state: _RunState, ctx: WorkflowContext[_RunState]) -> None:
        while state.calls and not state.clarification:
            state = self._execute_call(state)
        await ctx.send_message(state)

    def _execute_call(self, state: _RunState) -> _RunState:
        if not state.calls:
            raise ValueError("Expected a tool call.")
        call = state.calls[0]
        if call.name not in self._tools or not call.call_id:
            raise ValueError("Model requested an unknown or disabled tool.")
        args = call.arguments
        allowed = {"first_name", "last_name"} if call.name == "query_driver_by_name" else {"number_text", "state"}
        if not isinstance(args, Mapping) or set(args) - allowed:
            raise ValueError("Invalid tool arguments.")
        parameters = dict(args)
        if call.name == "query_driver_by_name":
            try:
                result = _extract_driver_name(args, state.source)
            except ValueError as exc:
                clarification = (
                    f"I couldn't determine the driver's name: {exc} "
                    "Please resend your request with the driver's first name and last name."
                )
                return replace(
                    state, messages=[*state.messages, Message("assistant", [Content.from_text(clarification)])],
                    calls=[], clarification=clarification,
                )
        else:
            number = args.get("number_text")
            try:
                if not isinstance(number, str) or not number.strip():
                    raise ValueError("The number is missing.")
                _assert_in_source(number, state.source, "number_text")
                _assert_complete_number(number, state.source, self._alphabet)
                normalised = _normalise_number(number, self._alphabet)
            except ValueError as exc:
                license_type = "license plate" if call.name == "query_license_plate" else "driver license"
                clarification = (
                    f"I couldn't determine the {license_type} number: {exc} "
                    f"Please resend your request with the {license_type} number, "
                    "using the exact characters or spoken words."
                )
                return replace(
                    state, messages=[*state.messages, Message("assistant", [Content.from_text(clarification)])],
                    calls=[], clarification=clarification,
                )
            region = parameters.get("state")
            source_region = _region_from_request(state.source)
            if region is not None:
                if not isinstance(region, str):
                    raise ValueError("State must be a string or null.")
                if not re.fullmatch(r"[^\W\d_]+(?:[ .'-][^\W\d_]+)*", region.strip()):
                    raise ValueError("State must contain letters, spaces, apostrophes, hyphens, or periods.")
                if not re.search(
                    r"(?<!\w)" + re.escape(region.strip()) + r"(?!\w)",
                    state.source,
                    re.IGNORECASE,
                ):
                    try:
                        resolved_region = resolve_us_region(region)
                        mentioned = us_region_is_mentioned(resolved_region, state.source)
                    except ValueError:
                        _assert_in_source(region, state.source, "state")
                        raise
                    if not mentioned:
                        _assert_in_source(region, state.source, "state")
                    region = resolved_region
            elif source_region is not None:
                region = source_region
            try:
                if region is not None:
                    region = resolve_us_region(region)
                    parameters["state"] = region
            except ValueError as exc:
                clarification = (
                    f"I couldn't determine the state: {exc} "
                    "Please provide a state abbreviation or an unambiguous supported city."
                )
                return replace(
                    state, messages=[*state.messages, Message("assistant", [Content.from_text(clarification)])],
                    calls=[], clarification=clarification,
                )
            if call.name == "query_license_plate":
                result = query_license_plate(normalised, region)
            else:
                result = query_driver_license(normalised, region)
        tool_message = Message("tool", [Content.from_function_result(
            call.call_id, result=result
        )])
        return replace(
            state, results=[*state.results, {"tool": call.name, "parameters": parameters}],
            messages=[*state.messages, tool_message], calls=state.calls[1:],
        )


class _OutputExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="output")

    @handler
    async def handle(self, state: _RunState, ctx: WorkflowContext[None, str]) -> None:
        if state.clarification:
            output = state.clarification
        elif state.results:
            output = json.dumps(state.results, ensure_ascii=False)
        elif state.answer:
            output = state.answer
        else:
            raise ValueError("Agent returned no answer.")
        await ctx.yield_output(output)


class AgentGraph:
    """Configured local license and driver-name extraction workflow."""

    def __init__(self, backend: ToolBackend, config_path: str | Path | None = None) -> None:
        self._backend = backend
        self.load_config(config_path or Path(__file__).with_name("config.json"))

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
        builder.add_edge(agent, tool, condition=lambda state: bool(state.calls))
        builder.add_edge(agent, output, condition=lambda state: not state.calls)
        builder.add_edge(tool, agent, condition=lambda state: not state.clarification)
        builder.add_edge(tool, output, condition=lambda state: bool(state.clarification))
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
