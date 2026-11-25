# Agent Graph

The `AgentGraph` class implements a configurable, local workflow for extracting license plates and driver license numbers from user messages using tool-augmented language models.

## Overview

This module provides a production-ready agent that:
- Parses user messages to identify license plate or driver license numbers
- Supports NATO phonetic alphabet and digit word conversion (e.g., "alfa" → "A", "one" → "1")
- Validates extracted numbers against the original message to prevent hallucination
- Enforces safety limits on tool calls to prevent infinite loops
- Handles multiple numbers in a single message through iterative tool invocation

## Architecture

The workflow is built using the [Agent Framework](https://github.com/) and consists of four main components:

### 1. `_AgentExecutor`
**Role**: Orchestrates the conversation flow and manages tool call limits.

- Sends messages to the local chat client
- Enforces the `max_tool_calls` limit to prevent runaway execution
- Routes to the `_ToolExecutor` when tool calls are present
- Routes to the `_OutputExecutor` when no further tool calls are needed

### 2. `LocalChatClient`
**Role**: Translates between Agent Framework messages and the Aion tool protocol.

- Converts internal `Message` objects to the Aion format
- Parses tool calls from the model's response using `<|tool_call|>` markers
- Extracts both text responses and function calls from a single model reply
- Validates JSON structure of tool call arguments

### 3. `_ToolExecutor`
**Role**: Executes tool calls and validates their arguments.

**Validation performed:**
- Tool name is known and enabled
- Arguments match the expected schema (`number_text`, optional `state`)
- Number text appears verbatim in the original user message
- Number occurs exactly once (not duplicated or ambiguous)
- Number components are complete (not truncated)
- Normalizes numbers using the NATO alphabet and digit word mapping

**On validation failure:**
- Generates a clarification request asking the user to resend with exact characters/words

### 4. `_OutputExecutor`
**Role**: Produces the final output.

Priority order:
1. Clarification message (if validation failed)
2. JSON array of extracted results (if tools were called)
3. Conversational response (if no tools were needed)

## Configuration

The agent is configured via a JSON file (default: `tools.json`):

```json
{
  "tools": ["query_license_plate", "query_driver_license"],
  "alphabet_file": "../tools/nato_alphabet.json",
  "max_tool_calls": 3
}
```

### Configuration Fields

| Field | Type | Description |
|-------|------|-------------|
| `tools` | `string[]` | List of enabled tools. Must be unique and from known tools: `query_license_plate`, `query_driver_license` |
| `alphabet_file` | `string` | Path to a JSON file mapping phonetic words to letters (e.g., `{"alfa": "A", "bravo": "B"}`) |
| `max_tool_calls` | `int` | Maximum number of tool calls allowed per message. Must be a positive integer (≥1) |

### Validation Rules

- `tools`: Must be a non-empty list of unique, known tool names
- `alphabet_file`: Must be a non-empty string path
- `max_tool_calls`: Must be a positive integer (booleans are rejected)
- Alphabet file: Must be a non-empty JSON object with lowercase word keys (letters/hyphens only) and single uppercase letter values

## Workflow Diagram

```
User Message
    │
    ▼
┌─────────────────┐
│  AgentExecutor  │ ──(no tool calls)──► OutputExecutor
│  (sends to LLM) │
└────────┬────────┘
         │
         │ (tool calls present)
         ▼
┌─────────────────┐
│  ToolExecutor   │ ──(clarification needed)──► OutputExecutor
│  (validates &   │
│   executes)     │ ──(success)───────────────► AgentExecutor (next iteration)
└─────────────────┘
```

## Message Flow Example

### Example 1: Single License Plate

**Input:**
```
"Plate alfa bravo charlie 123 in California"
```

**Flow:**
1. AgentExecutor → LLM → Tool call: `query_license_plate(number_text="alfa bravo charlie 123", state="California")`
2. ToolExecutor validates and normalizes → `ABC123`
3. ToolExecutor → AgentExecutor (no more calls) → OutputExecutor
4. **Output:** `[{"license_plate_number": "ABC123", "state": "California"}]`

### Example 2: Multiple Plates

**Input:**
```
"2 plates DEF456 and alice boy david xray 222"
```

**Flow:**
1. AgentExecutor → LLM → Tool call: `query_license_plate(number_text="DEF456")`
2. ToolExecutor → AgentExecutor
3. AgentExecutor → LLM → Tool call: `query_license_plate(number_text="alice boy david xray 222")`
4. ToolExecutor normalizes → `ABDX222`
5. ToolExecutor → AgentExecutor → OutputExecutor
6. **Output:** `[{"license_plate_number": "DEF456"}, {"license_plate_number": "ABDX222"}]`

### Example 3: Ambiguous Number

**Input:**
```
"Plate abc"
```

**Flow:**
1. AgentExecutor → LLM → Tool call: `query_license_plate(number_text="abc")`
2. ToolExecutor validates → "Ambiguous number component: 'abc'" (not in alphabet, not a digit, not mixed alphanumeric)
3. ToolExecutor generates clarification
4. ToolExecutor → OutputExecutor
5. **Output:** `"I couldn't determine the license plate number: Ambiguous number component: 'abc'. Please resend your request with the license plate number, using the exact characters or spoken words."`

## System Instructions

The agent operates under these constraints:

- Use tools **only** for license plate or driver license extraction
- For unrelated messages, respond conversationally without calling tools
- Pass `number_text` as the **exact** words/characters from the user's message (not inferred or rewritten)
- Call tools **separately** for each number (even multiple of the same type)
- Pass `state` only if explicitly mentioned
- Ask for clarification if information is missing or ambiguous
- **Never invent information**

## Safety Features

### Tool Call Limiting

The `max_tool_calls` parameter prevents:
- Infinite loops from repeated tool calls
- Resource exhaustion
- Runaway costs

The workflow's `max_iterations` is set to `2 * max_tool_calls + 3` to provide sufficient turns for tool execution and response handling.

### Input Validation

- Numbers must appear **exactly once** in the source message
- Adjacent alphanumeric sequences trigger clarification requests
- Numbers must contain only letters, digits, spaces, or hyphens
- All components must be unambiguous (either in alphabet, digits, or mixed alphanumeric)

### Schema Validation

- Tool arguments are strictly validated against the schema
- Unknown tools are rejected
- Invalid argument types raise errors

## Usage

```python
from core.graph import AgentGraph
from your_backend import YourToolBackend

backend = YourToolBackend()
graph = AgentGraph(backend, "path/to/config.json")

result = await graph.run("Extract plate alfa bravo 123")
print(result)
```

## Error Handling

The agent raises `ValueError` for:
- Invalid configuration files
- Invalid NATO alphabet format
- Invalid tool arguments
- Numbers not found in source message
- Duplicate or ambiguous numbers

The agent raises `RuntimeError` for:
- Workflow execution failures
- Missing outputs

## Files

- [`graph.py`](../core/graph.py) - Main agent implementation
- [`tools.json`](../core/tools.json) - Default configuration
- [`nato_alphabet.json`](../tools/nato_alphabet.json) - NATO phonetic alphabet mapping
