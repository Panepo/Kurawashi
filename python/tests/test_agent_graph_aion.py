from __future__ import annotations

import asyncio
import json
import os

import pytest

from core.graph import AgentGraph
from core.llm import LLMInference


@pytest.mark.skipif(
    os.environ.get("RUN_AION_AGENT_SMOKE") != "1",
    reason="Set RUN_AION_AGENT_SMOKE=1 to run the local model smoke test.",
)
def test_real_aion_tool_calls() -> None:
    graph = AgentGraph(LLMInference("aion", device="webgpu"))
    for request in ("driver license", "license plate"):
        clarification = asyncio.run(graph.run(request))
        assert request in clarification.lower()
        assert "number" in clarification.lower()
    plate = asyncio.run(graph.run("Query license plate alfa bravo charlie 123 in California."))
    plate_without_state = asyncio.run(graph.run("Query license plate alfa bravo charlie 123."))
    driver = asyncio.run(graph.run("Query driver license D1234567."))
    spoken_driver = asyncio.run(graph.run("Query driver license delta one two three."))
    two_plates = asyncio.run(graph.run("2 plate ABC123 and alice boy david xray 222"))
    assert json.loads(plate) == [{"license_plate_number": "ABC123", "state": "California"}]
    assert json.loads(plate_without_state) == [{"license_plate_number": "ABC123", "state": None}]
    assert json.loads(driver) == [{"driver_license_number": "D1234567"}]
    assert json.loads(spoken_driver) == [{"driver_license_number": "D123"}]
    assert json.loads(two_plates) == [
        {"license_plate_number": "ABC123", "state": None},
        {"license_plate_number": "ABDX222", "state": None},
    ]
