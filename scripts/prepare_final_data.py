"""Deterministic larger splits; exclude qualification questions and inspected pilot test pool."""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset

from src.utils.persistence import write_json

ROOT = Path(__file__).resolve().parents[1]
REVISION = "740312add88f781978c0658806c59bc2815b9866"


def prepare() -> None:
    validation = [
        json.loads(s)
        for s in (ROOT / "data/gsm8k_phase2_validation.jsonl").read_text().splitlines()
    ]
    inspected = [
        json.loads(s) for s in (ROOT / "data/gsm8k_phase2_test.jsonl").read_text().splitlines()
    ]
    exclusions = {
        "train": {r["question"] for r in validation},
        "test": {r["question"] for r in inspected},
    }
    manifest = {
        "dataset": "openai/gsm8k",
        "revision": REVISION,
        "shuffle_seed": 20261005,
        "validation": "data/gsm8k_phase2_validation.jsonl",
        "splits": {},
    }
    selected = {}
    for split, limit in [("train", 256), ("test", 128)]:
        dataset = load_dataset("openai/gsm8k", "main", revision=REVISION, split=split)
        rows = []
        for row in dataset.shuffle(seed=20261005):
            if row["question"] in exclusions[split]:
                continue
            key = hashlib.sha256(row["question"].encode()).hexdigest()[:16]
            rows.append(
                {
                    "id": f"gsm8k-final-{split}-{key}",
                    "question": row["question"],
                    "reference": row["answer"].rsplit("####", 1)[-1].strip(),
                    "metadata": {"source_split": split},
                }
            )
            if len(rows) == limit:
                break
        assert len(rows) == limit
        target = ROOT / f"data/gsm8k_final_{split}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        manifest["splits"][split] = {
            "path": str(target.relative_to(ROOT)),
            "count": limit,
            "excluded_questions": len(exclusions[split]),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        selected[split] = {r["question"] for r in rows}
    assert not selected["train"] & selected["test"]
    assert not selected["train"] & {r["question"] for r in validation}
    write_json(ROOT / "experiments/final/dataset_manifest.json", manifest)


if __name__ == "__main__":
    prepare()
