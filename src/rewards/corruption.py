"""Order-independent seeded corruption, keyed by example and generated response."""

import hashlib
import json
import random
from dataclasses import dataclass

from src.data.schema import Example
from src.rewards.verifiers import Verifier, VerifierResult


@dataclass(frozen=True)
class CorruptionConfig:
    kind: str = "flip"
    probability: float = 0.0
    seed: int = 0
    noise_std: float = 0.1
    confidence_scale: float = 1.0
    bias_key: str = "class"
    bias_value: str = "addition"
    bias_reward: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {
            "flip",
            "false_positive",
            "false_negative",
            "missing",
            "noise",
            "bias",
        }:
            raise ValueError(f"Unknown corruption: {self.kind}")
        for name in ("probability", "confidence_scale", "bias_reward"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if not 0 <= self.noise_std < float("inf"):
            raise ValueError("noise_std must be finite and nonnegative")


class CorruptedVerifier:
    def __init__(self, verifier: Verifier, config: CorruptionConfig):
        self.verifier = verifier
        self.config = config

    def verify(self, example: Example, response: str) -> VerifierResult:
        base = self.verifier.verify(example, response)
        cfg = self.config
        key = json.dumps([cfg.seed, example.id, response], ensure_ascii=False).encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
        reward = base.reward
        eligible = cfg.kind != "bias" or str(example.metadata.get(cfg.bias_key)) == cfg.bias_value
        applied = reward is not None and eligible and rng.random() < cfg.probability
        if applied:
            assert reward is not None
            if cfg.kind == "flip":
                reward = 1 - reward
            elif cfg.kind == "false_positive" and reward < 0.5:
                reward = 1.0
            elif cfg.kind == "false_negative" and reward >= 0.5:
                reward = 0.0
            elif cfg.kind == "missing":
                reward = None
            elif cfg.kind == "noise":
                reward = max(0.0, min(1.0, reward + rng.gauss(0, cfg.noise_std)))
            elif cfg.kind == "bias":
                reward = cfg.bias_reward
        # Degradation is deliberately independent of whether a particular flip occurred.
        confidence = 0.0 if reward is None else base.confidence * cfg.confidence_scale
        return VerifierResult(
            reward,
            confidence,
            {
                "corruption": cfg.kind,
                "applied": applied,
                "base": base.metadata,
            },
        )
