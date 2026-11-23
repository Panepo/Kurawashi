from __future__ import annotations

import argparse
import subprocess  # nosec B404
from pathlib import Path
from typing import NamedTuple


MODELS_DIR = Path(__file__).parent.parent / "models"


class ModelEntry(NamedTuple):
    folder: str
    url: str

class VisonModelEntry(NamedTuple):
    folder: str
    url: str

class EmbeddingModelEntry(NamedTuple):
    folder: str
    url: str
    dim: int

MODELS: dict[str, ModelEntry] = {
    "aion":     ModelEntry("aion-instruct",                        "https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B-ONNX-WebGPU"),
    "me5s":     EmbeddingModelEntry("multilingual-e5-small",       "https://huggingface.co/intfloat/multilingual-e5-small", 384),
    "me5b":     EmbeddingModelEntry("multilingual-e5-base",        "https://huggingface.co/intfloat/multilingual-e5-base", 768),
    "me5l":     EmbeddingModelEntry("multilingual-e5-large",       "https://huggingface.co/intfloat/multilingual-e5-large", 1024),
}


def download_model(model: str) -> None:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(MODELS)}")

    entry = MODELS[model]

    MODELS_DIR.mkdir(exist_ok=True)
    target_dir = MODELS_DIR / entry.folder

    if target_dir.exists():
        print(f"Model '{model}' already present at {target_dir}")
        return

    print(f"Downloading '{model}' from {entry.url} ...")
    subprocess.run(["git", "clone", entry.url, str(target_dir)], check=True)  # nosec B603 B607


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=list(MODELS),
        help=f"Model to download. Choices: {', '.join(MODELS)}",
    )
    args = parser.parse_args()
    download_model(args.model)
