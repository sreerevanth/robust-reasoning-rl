"""One verifier construction path shared by training and evaluation."""

from typing import Any

from src.rewards.corruption import CorruptedVerifier, CorruptionConfig
from src.rewards.ensemble import Ensemble
from src.rewards.verifiers import ExactVerifier, RuleVerifier, Verifier


def build_verifier(config: dict[str, Any]) -> Verifier:
    kind = config.get("kind", "rule")
    if kind not in {"exact", "rule"}:
        raise ValueError(f"Unknown verifier: {kind}")
    verifier: Verifier = ExactVerifier() if kind == "exact" else RuleVerifier()
    if "corruption" in config:
        verifier = CorruptedVerifier(verifier, CorruptionConfig(**config["corruption"]))
    return verifier


def build_ensemble(config: dict[str, Any]) -> Ensemble:
    return Ensemble(
        [build_verifier(member) for member in config.get("members", [{"kind": "rule"}])],
        config.get("strategy", "mean"),
    )
