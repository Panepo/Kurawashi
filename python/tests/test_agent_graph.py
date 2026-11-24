from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_framework import Message

from core.graph import AgentGraph, LocalChatClient, _normalise_number
from core.llm import LLMInference


class FakeBackend:
    def __init__(self, replies: list[str]) -> None:
        self.replies = iter(replies)
        self.messages: list[list[dict[str, object]]] = []
        self.tools: list[dict[str, object]] = []

    def chat_with_tools(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]],
        max_new_tokens: int = 512, temperature: float = 0,
    ) -> str:
        self.messages.append(messages)
        self.tools = tools
        return next(self.replies)


def call(name: str, arguments: dict[str, object]) -> str:
    return "<|tool_call|>" + json.dumps({
        "name": name, "arguments": arguments,
    }) + "<|/tool_call|>"


@pytest.mark.parametrize(
    ("name", "message", "arguments", "expected"),
    [
        (
            "query_license_plate", "Plate alfa bravo charlie 123 in California",
            {"number_text": "alfa bravo charlie 123", "state": "California"},
            {"license_plate_number": "ABC123", "state": "California"},
        ),
        (
            "query_license_plate", "Plate alfa bravo charlie 123",
            {"number_text": "alfa bravo charlie 123"},
            {"license_plate_number": "ABC123", "state": None},
        ),
        (
            "query_driver_license", "Driver license D1234567",
            {"number_text": "D1234567"},
            {"driver_license_number": "D1234567"},
        ),
    ],
)
def test_tool_round_trip(
    name: str, message: str, arguments: dict[str, object], expected: dict[str, object]
) -> None:
    backend = FakeBackend([call(name, arguments), "Done."])
    result = asyncio.run(AgentGraph(backend).run(message))
    assert json.loads(result) == expected
    assert {tool["name"] for tool in backend.tools} == {
        "query_license_plate", "query_driver_license"
    }
    assert backend.messages[1][-2]["tool_calls"] == [{"name": name, "arguments": arguments}]
    assert json.loads(str(backend.messages[1][-1]["content"])) == expected


def test_multiple_tool_calls_return_each_result() -> None:
    backend = FakeBackend([
        call("query_license_plate", {"number_text": "ABC123"}),
        call("query_driver_license", {"number_text": "D123"}),
        "Done.",
    ])
    result = asyncio.run(AgentGraph(backend).run("Plate ABC123, driver license D123"))
    assert json.loads(result) == [
        {"license_plate_number": "ABC123", "state": None},
        {"driver_license_number": "D123"},
    ]


def test_clarification_is_returned_without_a_tool_call() -> None:
    backend = FakeBackend(["Please provide the license number."])
    assert asyncio.run(AgentGraph(backend).run("Query a plate")) == (
        "Please provide the license number."
    )


@pytest.mark.parametrize(
    ("name", "message", "arguments", "expected"),
    [
        ("query_driver_license", "driver license", {}, "driver license"),
        ("query_driver_license", "driver license", {"number_text": None}, "driver license"),
        ("query_license_plate", "plate", {"number_text": "  "}, "license plate"),
        ("query_driver_license", "driver license", {"number_text": "driver license"}, "driver license"),
        ("query_license_plate", "license plate", {"number_text": "plate alfa bravo charlie 123"}, "license plate"),
    ],
)
def test_missing_number_requests_clarification(
    name: str, message: str, arguments: dict[str, object], expected: str,
) -> None:
    backend = FakeBackend([call(name, arguments)])
    result = asyncio.run(AgentGraph(backend).run(message))
    assert expected in result.lower()
    assert "number" in result.lower()
    assert len(backend.messages) == 1


def test_missing_number_after_result_requests_clarification() -> None:
    backend = FakeBackend([
        call("query_license_plate", {"number_text": "ABC123"}),
        call("query_driver_license", {}),
    ])
    result = asyncio.run(AgentGraph(backend).run("Plate ABC123 and driver license"))
    assert "driver license" in result.lower()
    assert "number" in result.lower()
    assert len(backend.messages) == 2


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (call("query_license_plate", {"number_text": "ABC123", "state": "Nevada"}), "Plate ABC123"),
        (call("query_license_plate", {"number_text": "ABC123", "state": "123"}), "Plate ABC123 in 123"),
        (call("query_driver_license", {"number_text": "D123", "state": "Ohio"}), "Driver D123 in Ohio"),
        (call("unregistered", {"number_text": "ABC123"}), "Plate ABC123"),
    ],
)
def test_invalid_or_invented_tool_arguments_fail(reply: str, message: str) -> None:
    with pytest.raises(ValueError):
        asyncio.run(AgentGraph(FakeBackend([reply])).run(message))


def test_malformed_tool_call_is_rejected() -> None:
    client = LocalChatClient(FakeBackend(["<|tool_call|>{bad}<|/tool_call|>"]), [])
    with pytest.raises(ValueError, match="invalid tool-call JSON"):
        asyncio.run(client.get_response([Message("user", ["plate"])]))


def test_tool_call_limit_is_enforced(tmp_path: Path) -> None:
    (tmp_path / "alphabet.json").write_text('{"alfa":"A"}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "tools": ["query_license_plate"], "alphabet_file": "alphabet.json", "max_tool_calls": 1,
    }), encoding="utf-8")
    backend = FakeBackend([
        call("query_license_plate", {"number_text": "alfa 123"}),
        call("query_license_plate", {"number_text": "alfa 123"}),
    ])
    with pytest.raises(ValueError, match="Maximum tool-call count exceeded"):
        asyncio.run(AgentGraph(backend, config).run("Plate alfa 123"))


def test_editable_alphabet_and_disabled_tool(tmp_path: Path) -> None:
    (tmp_path / "alphabet.json").write_text('{"custom":"Z"}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "tools": ["query_license_plate"], "alphabet_file": "alphabet.json", "max_tool_calls": 2,
    }), encoding="utf-8")
    backend = FakeBackend([call("query_license_plate", {"number_text": "custom 123"}), "Done."])
    assert json.loads(asyncio.run(AgentGraph(backend, config).run("Plate custom 123"))) == {
        "license_plate_number": "Z123", "state": None,
    }
    assert [tool["name"] for tool in backend.tools] == ["query_license_plate"]


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "tools": ["arbitrary_tool"], "alphabet_file": "alphabet.json", "max_tool_calls": 2,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unique, known tools"):
        AgentGraph(FakeBackend([]), config)


def test_spoken_digits_and_xray() -> None:
    assert _normalise_number("x-ray one two 003", {"x-ray": "X"}) == "X12003"


@pytest.mark.parametrize(
    ("name", "number", "message", "reason"),
    [
        ("query_license_plate", "ABC123", "Plate XYZ123", "verbatim"),
        ("query_driver_license", "unknown words", "Driver license unknown words", "Ambiguous"),
        ("query_license_plate", "alfa bravo charlie", "Plate alfa bravo charlie 123", "incomplete"),
    ],
)
def test_invalid_number_requests_clarification(
    name: str, number: str, message: str, reason: str,
) -> None:
    backend = FakeBackend([call(name, {"number_text": number})])
    result = asyncio.run(AgentGraph(backend).run(message))
    assert reason in result
    assert "number" in result.lower()


def test_llm_passes_tools_and_structured_messages_to_template() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages: str, **kwargs: object) -> str:
            assert json.loads(messages) == [{"role": "user", "content": "plate ABC123"}]
            assert json.loads(str(kwargs["tools"])) == [{"name": "query_license_plate"}]
            assert kwargs["add_generation_prompt"] is True
            return "rendered prompt"

    llm = LLMInference.__new__(LLMInference)
    llm._chat_template = "template"
    with patch.object(llm, "_tokenizer", Tokenizer(), create=True), patch.object(
        llm, "generate", return_value="response"
    ) as generate:
        assert llm.chat_with_tools(
            [{"role": "user", "content": "plate ABC123"}], [{"name": "query_license_plate"}]
        ) == "response"
        generate.assert_called_once_with("rendered prompt", max_new_tokens=512, temperature=0)
