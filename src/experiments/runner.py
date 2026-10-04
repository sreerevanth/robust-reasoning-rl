"""Full sweeps train actual policies; fixture sweeps only exercise verifier plumbing."""

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loading import load_dataset
from src.evaluation.runner import evaluate
from src.training.runner import train
from src.utils.logging import event
from src.utils.persistence import provenance, write_json


def experiment(config: dict[str, Any]) -> dict[str, Any]:
    exp = config.get("experiment", {})
    mode = exp.get("mode", "fixture")
    if mode not in {"fixture", "train"}:
        raise ValueError("Experiment mode must be fixture or train")
    levels = exp.get("levels", [0, .1, .2, .4, .6])
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
    failures = []
    manifest = provenance(config, execution_kind="training_sweep" if mode == "train" else "software_fixture_sweep")
    write_json(output / "manifest.json", manifest)
    for seed in seeds:
        for level in levels:
            for method in methods:
                run = copy.deepcopy(config)
                run["seed"] = seed
                run_dir = output / f"seed-{seed}" / f"level-{level:g}" / method
                run["output_dir"] = str(run_dir / "evaluation")
                for index, member in enumerate(run["training_verifier"]["members"]):
                    member["corruption"] = {**exp.get("corruption", {"kind": "flip"}),
                                            "probability": level, "seed": seed + index * 1009}
                try:
                    training_result = None
                    if mode == "train" and method != "base":
                        train_config = copy.deepcopy(run)
                        train_config["dataset"] = exp["train_dataset"]
                        train_config["output_dir"] = str(run_dir / "training")
                        train_config["reward"] = {"strategy": "standard"} if method == "baseline" else exp.get(
                            "robust_reward", {"strategy": "combined"})
                        training_result = train(train_config)
                        if train_config["training"]["lora"]["enabled"]:
                            run["model"]["adapter"] = training_result["checkpoint"]
                        else:
                            run["model"]["name"] = training_result["checkpoint"]
                    payload = evaluate(run)
                    row = {"seed": seed, "method": method, "corruption_level": level,
                           "corruption_kind": exp.get("corruption", {}).get("kind", "flip"),
                           "execution_kind": manifest["execution_kind"], **payload["metrics"]}
                    if training_result:
                        row["train_loss"] = training_result["metrics"].get("train_loss")
                        row["training_steps"] = run["training"]["max_steps"]
                        row.update({f"training_{k}": v for k, v in training_result["reward_statistics"].items()})
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
                    write_json(output / "summary.json", {"metadata": manifest, "runs": rows, "failures": failures})
                    event("experiment_run_complete", method=method, seed=seed, corruption_level=level)
                except Exception as error:
                    failures.append({"seed": seed, "level": level, "method": method,
                                     "type": type(error).__name__, "message": str(error)})
                    write_json(output / "summary.json", {"metadata": manifest, "runs": rows, "failures": failures})
                    if not exp.get("continue_on_error", False):
                        raise
    return {"metadata": manifest, "runs": rows, "failures": failures}
