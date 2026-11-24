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

## License extraction agent

The Agent Framework workflow extracts license information from text or
transcribed speech; it does not perform an external lookup or audio
transcription. For example:

```powershell
python .\python\run_agent.py "Query license plate alfa bravo charlie 123 in California."
python .\python\run_agent.py "Query driver license D1234567."
```

Successful tool results are always returned in a JSON array, for example
`[{"license_plate_number": "ABC123", "state": "California"}]` or
`[{"driver_license_number": "D1234567"}]`. Multiple calls produce multiple
array entries in call order. An unspecified plate state is `null`. The agent
may instead ask for clarification. Invalid, invented, or potentially incomplete
numbers trigger a clarification rather than a guessed result; other invalid tool
arguments cause an error. Review results
before relying on them: model tool selection and extraction are not guaranteed,
and the checks cannot detect every omitted or incorrect value.

When a required number is missing, resend the request with the license type and
number; interactive agent requests are processed independently.

The configured tools and call limit are in `core\tools.json`; the editable NATO
alphabet mapping is in `core\nato_alphabet.json`. Use `--config` to select
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
