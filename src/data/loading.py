"""Centralized local JSONL and Hugging Face dataset adapters."""

import json
import random
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
    indexed_rows = list(enumerate(rows))
    if "shuffle_seed" in config:
        random.Random(config["shuffle_seed"]).shuffle(indexed_rows)
    offset = config.get("offset", 0)
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("Dataset offset must be a nonnegative integer")
    if "limit" in config and (not isinstance(config["limit"], int) or config["limit"] < 1):
        raise ValueError("Dataset limit must be a positive integer")
    indexed_rows = indexed_rows[offset:]
    examples = []
    for index, row in indexed_rows:
        answer = str(row[config.get("answer_column", "reference")])
        if config.get("answer_separator"):
            answer = answer.rsplit(config["answer_separator"], 1)[-1].strip()
        examples.append(
            Example(
                id=config.get("id_prefix", "") + str(row.get(config.get("id_column", "id"), index)),
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
