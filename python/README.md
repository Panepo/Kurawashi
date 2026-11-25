# Local Aion inference

With your Python environment active, run these commands from the repository root:

```powershell
python -m pip install -r .\python\requirements-webgpu.txt
python -m pip install --no-deps onnxruntime-genai==0.14.1
python -m pip install -r .\python\requirements-agent.txt
python -m pip install -r .\python\requirements-test.txt
python -m pytest .\python\tests\test_aion_inference.py
```

Install GenAI with `--no-deps` so it uses the WebGPU runtime rather than pulling
in the standard `onnxruntime` package. If the model is not already present under
`python\models\aion-instruct`, run `python .\python\scripts\model_download.py -m aion`
first (requires Git LFS for the model weights).

## Command-line runner

From the repository root:

```powershell
python .\python\run_llm.py --prompt "Explain what ONNX Runtime is."
python .\python\run_llm.py --stream --prompt "Write a haiku."
python .\python\run_llm.py --chat
```

The runner defaults to Aion on WebGPU. Use `--device cpu` to run on the CPU.
Interactive chat streams replies, keeps conversation history, and exits when
you type `exit`. Run `python .\python\run_llm.py --help` for all options.

## License and driver-name extraction agent

The Agent Framework workflow extracts license information and driver names from text or
transcribed speech; it does not perform an external lookup or audio
transcription. For example:

```powershell
python .\python\run_agent.py "Query license plate alfa bravo charlie 123 in California."
python .\python\run_agent.py "Query driver license D1234567."
python .\python\run_agent.py "Query driver John Smith."
```

Successful tool calls are returned in a JSON array with the tool name and
parameters, for example
`[{"tool": "query_driver_by_name", "parameters": {"first_name": "John", "last_name": "Smith"}}]`.
Multiple calls produce multiple array entries in call order. The agent
may instead ask for clarification. Invalid, invented, or potentially incomplete
numbers trigger a clarification rather than a guessed result; other invalid tool
arguments cause an error. Review results
before relying on them: model tool selection and extraction are not guaranteed,
and the checks cannot detect every omitted or incorrect value.
State names are returned as USPS abbreviations. The agent resolves a curated
offline set of U.S. city names to states (for example, Seattle to `WA`); unknown
or ambiguous cities require the user to provide a state.

The `query_driver_by_name` tool requires `first_name` and `last_name`; its call
record includes those parameters.
Names retain their spelling, including compound names, apostrophes, and hyphens.
Missing or invalid names trigger clarification; an ambiguous first/last name
split should also be clarified rather than guessed. This tool extracts names
only; it does not retrieve driver records.

When a required number is missing, resend the request with the license type and
number; interactive agent requests are processed independently.

The configured tools and call limit are in `core\tools.json`; the editable NATO
alphabet mapping is in `tools\nato_alphabet.json`. Use `--config` to select
another config file; its alphabet path is relative to that file. Tool names
must be in the built-in registry. Use `--device cpu` to select CPU inference.

Run the fast tests and, optionally, the real-model graph smoke tests with:

```powershell
python -m pytest .\python\tests\test_agent_graph.py
$env:RUN_AION_AGENT_SMOKE = "1"
python -m pytest .\python\tests\test_agent_graph_aion.py
Remove-Item Env:\RUN_AION_AGENT_SMOKE
```

## Smoke test

The smoke test sends system and user messages to the local model on WebGPU and
checks that the reply names Paris. Pytest shows the generated reply under
`Captured stdout call` even when the test passes. Add `-s` to see it as it is
printed instead.
