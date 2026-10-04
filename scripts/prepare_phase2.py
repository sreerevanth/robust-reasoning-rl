"""Download pinned public artifacts and freeze deterministic GSM8K subset selection."""

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import psutil
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download

from src.data.schema import Example
from src.utils.persistence import write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "experiments/phase2"
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_name = "openai/gsm8k"
    model_revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    dataset_revision = "740312add88f781978c0658806c59bc2815b9866"
    system_adapters = None
    if platform.system() == "Windows":
        detected = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json",
            ],
            text=True,
        )
        system_adapters = json.loads(detected)
    write_json(
        out / "hardware.json",
        {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "system_video_adapters": system_adapters,
            "gpus": [
                {
                    "name": torch.cuda.get_device_properties(i).name,
                    "vram_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ],
            "system_ram_bytes": psutil.virtual_memory().total,
            "available_ram_bytes": psutil.virtual_memory().available,
            "physical_cpus": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(),
        },
    )
    print(f"Pinned model={model_revision}, dataset={dataset_revision}", flush=True)
    snapshot_download(
        model_name,
        revision=model_revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    )
    dataset = load_dataset(dataset_name, "main", revision=dataset_revision)
    # Selection is fixed before observing any model outputs, without difficulty filtering.
    selection_seed = 20261004
    train_rows = dataset["train"].shuffle(seed=selection_seed)
    test_rows = dataset["test"].shuffle(seed=selection_seed)
    split_indices = {
        "train": (train_rows, range(32)),
        "validation": (train_rows, range(32, 48)),
        "test": (test_rows, range(32)),
    }
    manifest = {
        "model": model_name,
        "model_revision": model_revision,
        "dataset": dataset_name,
        "dataset_revision": dataset_revision,
        "subset": "main",
        "selection_seed": selection_seed,
        "selection": (
            "shuffle full official splits; first 32 train, next 16 validation, first 32 test"
        ),
        "preprocessing": "reference is stripped text after final ####; no difficulty filtering",
        "splits": {},
    }
    questions = set()
    for name, (rows, indices) in split_indices.items():
        path = root / f"data/gsm8k_phase2_{name}.jsonl"
        examples = []
        for index in indices:
            row = rows[index]
            digest = hashlib.sha256(row["question"].encode()).hexdigest()
            assert digest not in questions, "Question overlap across splits"
            questions.add(digest)
            examples.append(
                Example(
                    f"gsm8k-{name}-{digest[:16]}",
                    row["question"],
                    row["answer"].rsplit("####", 1)[-1].strip(),
                    {
                        "source": dataset_name,
                        "revision": dataset_revision,
                        "official_split": "test" if name == "test" else "train",
                        "selection_index": index,
                        "question_sha256": digest,
                    },
                )
            )
        path.write_text(
            "".join(json.dumps(asdict(e)) + "\n" for e in examples), encoding="utf-8", newline="\n"
        )
        manifest["splits"][name] = {
            "path": path.relative_to(root).as_posix(),
            "count": len(examples),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    write_json(out / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
