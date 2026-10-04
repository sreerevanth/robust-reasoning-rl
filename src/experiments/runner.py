"""Full sweeps train actual policies; fixture sweeps only exercise verifier plumbing."""

import copy
import gc
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loading import load_dataset
from src.evaluation.runner import evaluate
from src.models.generation import RecordedGenerator
from src.training.runner import train
from src.utils.logging import event
from src.utils.persistence import provenance, write_json


def experiment(config: dict[str, Any]) -> dict[str, Any]:
    exp = config.get("experiment", {})
    mode = exp.get("mode", "fixture")
    if mode not in {"fixture", "train"}:
        raise ValueError("Experiment mode must be fixture or train")
    levels = exp.get("levels", [0, 0.1, 0.2, 0.4, 0.6])
    if not levels or any(not 0 <= level <= 1 for level in levels):
        raise ValueError("Corruption levels must be nonempty and in [0,1]")
    seeds = exp.get("seeds", [config["seed"]])
    if not seeds or len(set(seeds)) != len(seeds) or len(set(levels)) != len(levels):
        raise ValueError("Seeds and corruption levels must be nonempty and unique")
    output = Path(config["output_dir"])
    if mode == "train":
        if config["model"]["backend"] != "huggingface":
            raise ValueError("Full experiments require real Hugging Face policies")
        if not exp.get("train_dataset"):
            raise ValueError("Full experiments require a separate train_dataset")
        train_examples = load_dataset(exp["train_dataset"])
        eval_examples = load_dataset(config["dataset"])
        if {e.id for e in train_examples} & {e.id for e in eval_examples}:
            raise ValueError("Training and evaluation IDs overlap")
        if {e.question for e in train_examples} & {e.question for e in eval_examples}:
            raise ValueError("Training and evaluation questions overlap")
    elif config["model"]["backend"] != "fixture":
        raise ValueError("Fixture sweeps must use the explicitly labelled fixture backend")
    methods = ["base", "baseline", "robust"] if mode == "train" else ["fixture_verifier_audit"]
    rows = []
    base_records: dict[int, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    manifest = provenance(
        config, execution_kind="training_sweep" if mode == "train" else "software_fixture_sweep"
    )
    write_json(output / "manifest.json", manifest)
    for seed in seeds:
        for level in levels:
            for method in methods:
                run = copy.deepcopy(config)
                run["seed"] = seed
                run_dir = output / f"seed-{seed}" / f"level-{level:g}" / method
                run["output_dir"] = str(run_dir / "evaluation")
                for index, member in enumerate(run["training_verifier"]["members"]):
                    member["corruption"] = {
                        **exp.get("corruption", {"kind": "flip"}),
                        "probability": level,
                        "seed": seed + index * 1009,
                    }
                try:
                    training_result = None
                    if mode == "train" and method != "base":
                        train_config = copy.deepcopy(run)
                        train_config["dataset"] = exp["train_dataset"]
                        train_config["output_dir"] = str(run_dir / "training")
                        train_config["reward"] = (
                            {"strategy": "standard"}
                            if method == "baseline"
                            else exp.get("robust_reward", {"strategy": "combined"})
                        )
                        training_result = train(train_config)
                        gc.collect()
                        if train_config["training"]["lora"]["enabled"]:
                            run["model"]["adapter"] = training_result["checkpoint"]
                        else:
                            run["model"]["name"] = training_result["checkpoint"]
                    if (
                        method == "base"
                        and exp.get("reuse_base_generations")
                        and seed in base_records
                    ):
                        payload = evaluate(run, RecordedGenerator(base_records[seed]))
                    else:
                        payload = evaluate(run)
                    if method == "base" and exp.get("reuse_base_generations"):
                        base_records[seed] = payload["records"]
                    gc.collect()
                    row = {
                        "seed": seed,
                        "method": method,
                        "corruption_level": level,
                        "corruption_kind": exp.get("corruption", {}).get("kind", "flip"),
                        "execution_kind": manifest["execution_kind"],
                        "experiment_label": exp.get("label"),
                        "evaluation_seconds": payload.get("metadata", {}).get("elapsed_seconds"),
                        **payload["metrics"],
                    }
                    if training_result:
                        row["train_loss"] = training_result["metrics"].get("train_loss")
                        row["training_steps"] = run["training"]["max_steps"]
                        row.update(
                            {
                                f"training_{k}": v
                                for k, v in training_result["reward_statistics"].items()
                            }
                        )
                        totals = training_result.get("reward_totals", {})
                        sample_count = training_result["reward_statistics"].get("samples", 0)
                        observed_count = totals.get("observed_samples", 0)
                        if observed_count:
                            row["training_observed_reward"] = (
                                totals["observed_reward_sum"] / observed_count
                            )
                            row["training_reward_variance"] = (
                                totals["observed_reward_square_sum"] / observed_count
                                - row["training_observed_reward"] ** 2
                            )
                        if sample_count:
                            for name, total in (
                                ("shaped_reward", "shaped_reward_sum"),
                                ("disagreement", "disagreement_sum"),
                                ("confidence", "confidence_sum"),
                                ("independent_accuracy", "independent_correct_sum"),
                                ("uncertainty_penalty", "uncertainty_penalty_sum"),
                            ):
                                if total in totals:
                                    row[f"training_{name}"] = totals[total] / sample_count
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
                    write_json(
                        output / "summary.json",
                        {"metadata": manifest, "runs": rows, "failures": failures},
                    )
                    event(
                        "experiment_run_complete", method=method, seed=seed, corruption_level=level
                    )
                except Exception as error:
                    failures.append(
                        {
                            "seed": seed,
                            "level": level,
                            "method": method,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    write_json(
                        output / "summary.json",
                        {"metadata": manifest, "runs": rows, "failures": failures},
                    )
                    if not exp.get("continue_on_error", False):
                        raise
    return {"metadata": manifest, "runs": rows, "failures": failures}
