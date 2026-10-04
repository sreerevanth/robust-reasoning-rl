"""Centralized local JSONL and Hugging Face dataset adapters."""

import json
from pathlib import Path
from typing import Any

from src.data.schema import Example


def load_dataset(config: dict[str, Any]) -> list[Example]:
    source = config.get("source", "local")
    if source == "local":
        with Path(config["path"]).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif source == "huggingface":
        from datasets import load_dataset as hf_load

        rows = hf_load(
            config["name"],
            config.get("subset"),
            split=config.get("split", "test"),
            revision=config.get("revision"),
        )
    else:
        raise ValueError(f"Unknown dataset source: {source}")
    examples = []
    for index, row in enumerate(rows):
        answer = str(row[config.get("answer_column", "reference")])
        if config.get("answer_separator"):
            answer = answer.rsplit(config["answer_separator"], 1)[-1].strip()
        examples.append(
            Example(
                id=str(row.get(config.get("id_column", "id"), index)),
                question=str(row[config.get("question_column", "question")]),
                reference=answer,
                metadata=dict(row.get("metadata", {})),
            )
        )
        if config.get("limit") and len(examples) >= config["limit"]:
            break
    if not examples or len({e.id for e in examples}) != len(examples):
        raise ValueError("Dataset must be nonempty and contain unique IDs")
    if any(not e.question.strip() or not e.reference.strip() for e in examples):
        raise ValueError("Questions and references must be nonempty")
    return examples
