# Local Aion inference

With your Python environment active, run these commands from the repository root:

```powershell
python -m pip install -r .\python\requirements-webgpu.txt
python -m pip install --no-deps onnxruntime-genai==0.14.1
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

## Smoke test

The smoke test sends system and user messages to the local model on WebGPU and
checks that the reply names Paris. Pytest shows the generated reply under
`Captured stdout call` even when the test passes. Add `-s` to see it as it is
printed instead.
