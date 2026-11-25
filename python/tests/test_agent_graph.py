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
            {"license_plate_number": "ABC123", "state": "CA"},
        ),
        (
            "query_license_plate", "Plate alfa bravo charlie 123",
            {"number_text": "alfa bravo charlie 123"},
            {"license_plate_number": "ABC123", "state": None},
        ),
        (
            "query_license_plate", "query WA license plate aaadsdas3231",
            {"number_text": "aaadsdas3231", "state": "WA"},
            {"license_plate_number": "AAADSDAS3231", "state": "WA"},
        ),
        (
            "query_license_plate", "query California license plate ABC123",
            {"number_text": "ABC123", "state": "CA"},
            {"license_plate_number": "ABC123", "state": "CA"},
        ),
        (
            "query_driver_license", "Driver license D1234567",
            {"number_text": "D1234567"},
            {"driver_license_number": "D1234567", "state": None},
        ),
        (
            "query_driver_license", "Driver license D1234567 in California",
            {"number_text": "D1234567", "state": "California"},
            {"driver_license_number": "D1234567", "state": "CA"},
        ),
        (
            "query_driver_by_name", "Query driver John Smith",
            {"first_name": "John", "last_name": "Smith"},
            {"first_name": "John", "last_name": "Smith"},
        ),
        (
            "query_driver_by_name", "Query driver Anne-Marie O'Neill",
            {"first_name": "Anne-Marie", "last_name": "O'Neill"},
            {"first_name": "Anne-Marie", "last_name": "O'Neill"},
        ),
        (
            "query_driver_by_name", "Driver first name Mary Jane, last name de la Cruz",
            {"first_name": "Mary Jane", "last_name": "de la Cruz"},
            {"first_name": "Mary Jane", "last_name": "de la Cruz"},
        ),
    ],
)
def test_tool_round_trip(
    name: str, message: str, arguments: dict[str, object], expected: dict[str, object]
) -> None:
    backend = FakeBackend([call(name, arguments), "Done."])
    result = asyncio.run(AgentGraph(backend).run(message))
    expected_parameters = dict(arguments)
    if "state" in expected_parameters and expected.get("state") is not None:
        expected_parameters["state"] = expected["state"]
    assert json.loads(result) == [{"tool": name, "parameters": expected_parameters}]
    assert {tool["name"] for tool in backend.tools} == {
        "query_license_plate", "query_driver_license", "query_driver_by_name"
    }
    driver_parameters = next(
        tool["parameters"] for tool in backend.tools if tool["name"] == "query_driver_license"
    )
    assert driver_parameters["required"] == ["number_text"]
    assert "state" in driver_parameters["properties"]
    assert backend.messages[1][-2]["tool_calls"] == [{"name": name, "arguments": arguments}]
    assert json.loads(str(backend.messages[1][-1]["content"])) == expected


def test_state_abbreviations_are_mentioned_in_tool_instructions() -> None:
    backend = FakeBackend([
        call("query_license_plate", {"number_text": "aaadsdas3231", "state": "WA"}),
        "Done.",
    ])
    asyncio.run(AgentGraph(backend).run("query WA license plate aaadsdas3231"))
    assert "standard two-letter U.S. state abbreviations" in str(
        backend.messages[0][0]["content"]
    )
    assert "query Seattle license plate" in str(backend.messages[0][0]["content"])


def test_city_is_resolved_when_model_omits_state() -> None:
    backend = FakeBackend([
        call("query_license_plate", {"number_text": "abc123"}),
        "Done.",
    ])
    result = asyncio.run(AgentGraph(backend).run("query seattle license plate abc123"))
    assert json.loads(result) == [{
        "tool": "query_license_plate",
        "parameters": {"number_text": "abc123", "state": "WA"},
    }]
    assert json.loads(str(backend.messages[1][-1]["content"])) == {
        "license_plate_number": "ABC123", "state": "WA",
    }


@pytest.mark.parametrize(
    ("city", "reason"),
    [("Portland", "ambiguous"), ("Everett", "not a supported")],
)
def test_ambiguous_or_unknown_city_requests_state(city: str, reason: str) -> None:
    backend = FakeBackend([call("query_license_plate", {"number_text": "ABC123"})])
    result = asyncio.run(AgentGraph(backend).run(
        f"query {city} license plate ABC123"
    ))
    assert reason in result.lower()
    assert "provide a state abbreviation" in result
    assert len(backend.messages) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"first_name": "John"},
        {"first_name": None, "last_name": "Smith"},
        {"first_name": "John", "last_name": "  "},
        {"first_name": "Jane", "last_name": "Smith"},
        {"first_name": "John", "last_name": "Smith123"},
    ],
)
def test_invalid_driver_name_requests_clarification(arguments: dict[str, object]) -> None:
    backend = FakeBackend([call("query_driver_by_name", arguments)])
    result = asyncio.run(AgentGraph(backend).run("Query driver John Smith"))
    assert "first name and last name" in result
    assert len(backend.messages) == 1


def test_driver_name_and_license_calls_in_one_batch() -> None:
    backend = FakeBackend([
        call("query_driver_by_name", {"first_name": "John", "last_name": "Smith"})
        + call("query_driver_license", {"number_text": "D123"}),
        "Done.",
    ])
    result = asyncio.run(AgentGraph(backend).run("Query driver John Smith and driver license D123"))
    assert json.loads(result) == [
        {
            "tool": "query_driver_by_name",
            "parameters": {"first_name": "John", "last_name": "Smith"},
        },
        {
            "tool": "query_driver_license",
            "parameters": {"number_text": "D123"},
        },
    ]
    assert len(backend.messages) == 2


def test_multiple_tool_calls_return_each_result() -> None:
    backend = FakeBackend([
        call("query_license_plate", {"number_text": "ABC123"}),
        call("query_driver_license", {"number_text": "D123"}),
        "Done.",
    ])
    result = asyncio.run(AgentGraph(backend).run("Plate ABC123, driver license D123"))
    assert json.loads(result) == [
        {"tool": "query_license_plate", "parameters": {"number_text": "ABC123"}},
        {"tool": "query_driver_license", "parameters": {"number_text": "D123"}},
    ]


def test_multiple_calls_in_one_response_are_processed_before_model_continues() -> None:
    first = {"number_text": "ABC123"}
    second = {"number_text": "alice boy david xray 222"}
    backend = FakeBackend([
        call("query_license_plate", first) + call("query_license_plate", second),
        "Done.",
    ])
    result = asyncio.run(AgentGraph(backend).run(
        "2 plate ABC123 and alice boy david xray 222"
    ))
    expected_output = [
        {"tool": "query_license_plate", "parameters": first},
        {"tool": "query_license_plate", "parameters": second},
    ]
    expected_results = [
        {"license_plate_number": "ABC123", "state": None},
        {"license_plate_number": "ABDX222", "state": None},
    ]
    assert json.loads(result) == expected_output
    assert len(backend.messages) == 2
    assert backend.messages[1][-3]["tool_calls"] == [
        {"name": "query_license_plate", "arguments": first},
        {"name": "query_license_plate", "arguments": second},
    ]
    assert [json.loads(str(message["content"])) for message in backend.messages[1][-2:]] == expected_results


def test_clarification_is_returned_without_a_tool_call() -> None:
    backend = FakeBackend(["Please provide the license number."])
    assert asyncio.run(AgentGraph(backend).run("Query a plate")) == (
        "Please provide the license number."
    )


def test_unrelated_message_gets_a_normal_response() -> None:
    response = "Good morning! How can I help?"
    backend = FakeBackend([response])

    assert asyncio.run(AgentGraph(backend).run("good morning")) == response
    assert "respond naturally and conversationally" in str(backend.messages[0][0]["content"])
    assert "tool_calls" not in backend.messages[0][-1]


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


def test_invalid_number_in_batch_requests_clarification() -> None:
    backend = FakeBackend([
        call("query_license_plate", {"number_text": "ABC123"})
        + call("query_driver_license", {}),
    ])
    result = asyncio.run(AgentGraph(backend).run("Plate ABC123 and driver license"))
    assert "driver license" in result.lower()
    assert "number" in result.lower()
    assert len(backend.messages) == 1


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (call("query_license_plate", {"number_text": "ABC123", "state": "Nevada"}), "Plate ABC123"),
        (call("query_license_plate", {"number_text": "ABC123", "state": "123"}), "Plate ABC123 in 123"),
        (call("query_driver_license", {"number_text": "D123", "state": "Ohio"}), "Driver D123"),
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


def test_tool_call_limit_applies_to_entire_batch(tmp_path: Path) -> None:
    (tmp_path / "alphabet.json").write_text('{"alfa":"A"}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "tools": ["query_license_plate"], "alphabet_file": "alphabet.json", "max_tool_calls": 1,
    }), encoding="utf-8")
    reply = call("query_license_plate", {"number_text": "alfa 123"})
    backend = FakeBackend([reply + reply])
    with pytest.raises(ValueError, match="Maximum tool-call count exceeded"):
        asyncio.run(AgentGraph(backend, config).run("Plate alfa 123"))
    assert len(backend.messages) == 1


def test_tool_call_limit_counts_prior_results_and_new_batch(tmp_path: Path) -> None:
    (tmp_path / "alphabet.json").write_text('{"alfa":"A"}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "tools": ["query_license_plate"], "alphabet_file": "alphabet.json", "max_tool_calls": 2,
    }), encoding="utf-8")
    reply = call("query_license_plate", {"number_text": "alfa 123"})
    backend = FakeBackend([reply, reply + reply])
    with pytest.raises(ValueError, match="Maximum tool-call count exceeded"):
        asyncio.run(AgentGraph(backend, config).run("Plate alfa 123"))


def test_editable_alphabet_and_disabled_tool(tmp_path: Path) -> None:
    (tmp_path / "alphabet.json").write_text('{"custom":"Z"}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "tools": ["query_license_plate"], "alphabet_file": "alphabet.json", "max_tool_calls": 2,
    }), encoding="utf-8")
    backend = FakeBackend([call("query_license_plate", {"number_text": "custom 123"}), "Done."])
    assert json.loads(asyncio.run(AgentGraph(backend, config).run("Plate custom 123"))) == [{
        "tool": "query_license_plate", "parameters": {"number_text": "custom 123"},
    }]
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
