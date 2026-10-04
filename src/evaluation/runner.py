"""Evaluate the policy with a reward source and a separate uncorrupted judge."""

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.data.loading import load_dataset
from src.data.schema import Generation
from src.evaluation.metrics import compute_metrics
from src.models.generation import Generator, build_generator
from src.rewards.answers import extract_answer
from src.rewards.factory import build_ensemble, build_verifier
from src.utils.config import validate
from src.utils.logging import event
from src.utils.persistence import provenance, write_json
from src.utils.reproducibility import seed_everything


def evaluate(config: dict[str, Any], generator: Generator | None = None) -> dict[str, Any]:
    validate(config)
    seed_everything(config["seed"])
    examples = load_dataset(config["dataset"])
    generator = generator or build_generator(config)
    reward_source = build_ensemble(config["training_verifier"])
    independent = build_verifier(config["evaluation_verifier"])
    records = []
    for example in examples:
        key = int.from_bytes(hashlib.sha256(example.id.encode()).digest()[:4], "big")
        outputs = generator.generate(
            example.question,
            config["generation"]["num_generations"],
            (config["seed"] + key) % (2**32),
        )
        if len(outputs) != config["generation"]["num_generations"]:
            raise ValueError("Generator returned an unexpected sample count")
        for response in outputs:
            parsed = extract_answer(response)
            observed = reward_source.verify(example, response)
            judged = independent.verify(example, response)
            records.append(
                Generation(
                    example.id,
                    response,
                    parsed.reasoning,
                    parsed.answer,
                    observed.to_dict(),
                    observed.reward,
                    observed.confidence,
                    observed.disagreement,
                    judged.reward == 1.0,
                    {"evaluation_verifier": asdict(judged)},
                )
            )
        event("evaluated_example", example_id=example.id, samples=len(outputs))
    metrics = compute_metrics(records, config["evaluation"]["k"])
    payload = {
        "metadata": provenance(
            config,
            execution_kind=(
                "software_fixture"
                if config["model"]["backend"] == "fixture"
                else "model_evaluation"
            ),
        ),
        "metrics": metrics,
        "records": [asdict(r) for r in records],
    }
    output = Path(config["output_dir"])
    write_json(output / "evaluation.json", payload)
    write_json(output / "config.json", config)
    with (output / "generations.jsonl").open("w", encoding="utf-8") as handle:
        import json

        for record in payload["records"]:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    event("evaluation_complete", output=str(output), **metrics)
    return payload
