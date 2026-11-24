# Agent graph plan

## Goal

Build a local LLM agent with Microsoft Agent Framework, using
[`LLMInference`](../python/core/llm.py) as its backend. Implement the graph
class in `python/core/graph.py`, including a function that loads a JSON file
defining the available tools.

The first tool, **query license plate**, extracts a license plate number and
an optional state from a user's message. The user can spell the plate with the
NATO phonetic alphabet: `alfa bravo charlie 123` means `ABC123`. Keep the
alphabet mapping in a separate editable JSON file.

The second tool, **query driver license**, extracts a driver license number
from the user's speech. Both tools return JSON and only extract information;
they do not look up a license plate or driver license. Initial input is text,
including speech transcribed elsewhere. Audio recognition is outside this plan.

Example successful tool outputs:

```json
{"license_plate_number": "ABC123", "state": null}
```

```json
{"driver_license_number": "D1234567"}
```

The plate output uses `null` when the state is not provided. If a number is
missing or ambiguous, report the problem or ask for clarification instead of
returning a guessed value.

## Proposed graph

```text
User message -> Aion agent -> tool call -> validate result -> response
                    ^            |
                    +-- result --+
```

- **Agent:** Aion interprets the user's request and chooses an available tool.
- **Router:** Use conditional routing in the workflow rather than a separate
  LLM agent.
- **Tool executor:** Resolve configured tools against a trusted registry,
  validate arguments, and execute the selected tool.
- **Verifier:** Use deterministic validation of tool inputs and JSON outputs.
  Do not invent a missing state or an ambiguous license number. Consider an LLM
  verifier only if evaluation shows that it improves results.
- **Supervisor:** Defer multi-agent supervision until a demonstrated use case
  requires it.

Aion is a candidate for the agent role, but its tool selection and extraction
quality must be evaluated. Microsoft notes that the small model can struggle
with complex multi-step reasoning; code should enforce validation and execution
limits independently of the model.

## Implementation steps

1. **Connect the backend.** Add an Agent Framework chat-client adapter around
   `LLMInference`. Its current interface accepts text messages and returns
   text, so support is needed for tool definitions, structured tool calls, and
   tool results. The local Aion chat template contains a tool-call format;
   verify the complete round trip with the installed ONNX Runtime GenAI version.
2. **Build the workflow.** Use Agent Framework's `WorkflowBuilder` in the
   graph class. Load and validate the JSON tool configuration and resolve
   configured names through a trusted registry, rather than importing or
   executing arbitrary code from configuration.
3. **Implement extraction tools.** For the plate tool, load the editable NATO
   alphabet mapping and normalize recognized letters and numbers into a plate.
   Return the optional state when it is present and unambiguous. For the driver
   license tool, extract the license number from the transcribed message. Both
   tools return JSON and surface ambiguity or invalid input instead of guessing.
4. **Add safeguards.** Validate configuration, tool arguments, and tool
   results; limit tool-call cycles; and report errors explicitly. Future tools
   with write operations should require user confirmation.
5. **Validate.** Add focused tests for both tools' JSON outputs, editable
   mappings, numbers, optional state, ambiguous input, invalid configuration,
   tool selection, and a complete tool-call round trip. Keep fast tests
   independent of the model using a fake backend, and add a separate local
   Aion smoke test.

## References

- [Agent Framework workflows](https://learn.microsoft.com/en-us/agent-framework/concepts/workflows/builder-and-execution)
- [Agent Framework custom chat clients](https://learn.microsoft.com/en-us/python/api/agent-framework-core/agent_framework.basechatclient?view=agent-framework-python-latest)
- [Aion model card](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B)
