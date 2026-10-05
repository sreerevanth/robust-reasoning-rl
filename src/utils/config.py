"""Validated YAML configuration; paths resolve relative to the project root."""

import copy
from pathlib import Path
from typing import Any

import yaml

from src.rewards.corruption import CorruptionConfig
from src.rewards.factory import build_ensemble, build_verifier
from src.rewards.shaping import RewardConfig

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "output_dir": "results/evaluation",
    "dataset": {"source": "local", "path": "data/tiny_eval.jsonl"},
    "model": {"backend": "fixture", "name": "HuggingFaceTB/SmolLM2-135M-Instruct"},
    "generation": {"num_generations": 4, "temperature": 0.7, "top_p": 0.95, "max_tokens": 128},
    "training_verifier": {"strategy": "mean", "members": [{"kind": "rule"}]},
    "evaluation_verifier": {"kind": "rule"},
    "reward": {"strategy": "standard"},
    "evaluation": {"k": [1, 2, 4]},
    "training": {
        "learning_rate": 5e-6,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_steps": 100,
        "save_steps": 50,
        "logging_steps": 5,
        "beta": 0.04,
        "gradient_checkpointing": True,
        "bf16": False,
        "fp16": False,
        "use_cpu": False,
        "lora": {
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": "all-linear",
        },
    },
}


def merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate(config: dict[str, Any]) -> None:
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    g = config["generation"]
    if config["model"].get("dtype", "float32") not in {"float32", "float16", "bfloat16"}:
        raise ValueError("model.dtype must be float32, float16, or bfloat16")
    if "top_k" in g and (not isinstance(g["top_k"], int) or g["top_k"] < 0):
        raise ValueError("generation.top_k must be a nonnegative integer")
    if not isinstance(g.get("sample_batch_size", 1), int) or g.get("sample_batch_size", 1) < 1:
        raise ValueError("sample_batch_size must be a positive integer")
    for key in ("num_generations", "max_tokens"):
        if not isinstance(g[key], int) or isinstance(g[key], bool) or g[key] < 1:
            raise ValueError(f"generation.{key} must be a positive integer")
    if not 0 <= g["temperature"] < float("inf") or not 0 < g["top_p"] <= 1:
        raise ValueError("Require finite temperature >= 0 and top_p in (0,1]")
    ks = config["evaluation"]["k"]
    if not ks or any(not isinstance(k, int) or not 1 <= k <= g["num_generations"] for k in ks):
        raise ValueError("Every evaluation k must be between 1 and num_generations")
    if config["model"]["backend"] not in {"fixture", "huggingface"}:
        raise ValueError("Unknown model backend")
    RewardConfig(**config["reward"])
    if config["evaluation_verifier"].get("corruption"):
        raise ValueError("Independent evaluation verifier must not be corrupted")
    build_verifier(config["evaluation_verifier"])
    build_ensemble(config["training_verifier"])
    for member in config["training_verifier"]["members"]:
        if member.get("corruption"):
            CorruptionConfig(**member["corruption"])
    t = config["training"]
    if t.get("fp16") and t["bf16"]:
        raise ValueError("fp16 and bf16 cannot both be enabled")
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "save_steps",
        "logging_steps",
    ):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if not 0 < t["learning_rate"] < float("inf") or not 0 <= t["beta"] < float("inf"):
        raise ValueError("Invalid learning rate or KL coefficient")
    lora = t["lora"]
    if lora["r"] < 1 or lora["alpha"] < 1 or not 0 <= lora["dropout"] < 1:
        raise ValueError("LoRA requires positive rank/alpha and dropout in [0,1)")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(loaded) - set(DEFAULTS) - {"experiment"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    config = merge(DEFAULTS, loaded)
    root = next((p for p in path.parents if (p / "pyproject.toml").exists()), path.parent)
    for section in ("dataset",):
        if config[section].get("path"):
            config[section]["path"] = str((root / config[section]["path"]).resolve())
    config["output_dir"] = str((root / config["output_dir"]).resolve())
    exp = config.get("experiment", {})
    if exp.get("train_dataset", {}).get("path"):
        exp["train_dataset"]["path"] = str((root / exp["train_dataset"]["path"]).resolve())
    if exp.get("qualification_dataset", {}).get("path"):
        exp["qualification_dataset"]["path"] = str(
            (root / exp["qualification_dataset"]["path"]).resolve()
        )
    if config["model"].get("adapter"):
        config["model"]["adapter"] = str((root / config["model"]["adapter"]).resolve())
    validate(config)
    return config
