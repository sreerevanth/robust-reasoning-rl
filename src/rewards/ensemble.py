"""Aggregations retain member outputs and availability, not only a scalar reward."""

import math
from dataclasses import asdict, dataclass

import numpy as np

from src.data.schema import Example
from src.rewards.verifiers import Verifier, VerifierResult


@dataclass(frozen=True)
class EnsembleResult:
    reward: float | None
    confidence: float
    disagreement: float
    variance: float
    vote_entropy: float
    confidence_dispersion: float
    members: list[VerifierResult]

    def to_dict(self) -> dict:
        return asdict(self)


class Ensemble:
    def __init__(self, verifiers: list[Verifier], strategy: str = "mean"):
        if not verifiers:
            raise ValueError("Ensemble needs at least one verifier")
        if strategy not in {"mean", "majority", "confidence_weighted", "minimum"}:
            raise ValueError(f"Unknown aggregation: {strategy}")
        self.verifiers, self.strategy = verifiers, strategy

    def verify(self, example: Example, response: str) -> EnsembleResult:
        members = [v.verify(example, response) for v in self.verifiers]
        available = [m for m in members if m.reward is not None]
        if not available:
            return EnsembleResult(None, 0, 0, 0, 0, 0, members)
        values = np.array([m.reward for m in available], dtype=float)
        weights = np.array([m.confidence for m in available])
        variance = float(np.var(values))
        disagreement = min(1.0, 4 * variance)
        positive = float(np.mean(values >= 0.5))
        entropy = -sum(p * math.log2(p) for p in (positive, 1 - positive) if p > 0)
        reward = float(np.mean(values))
        if self.strategy == "majority":
            # A tie gets 0.5, preserving explicit uncertainty.
            reward = float(positive > 0.5) if positive != 0.5 else 0.5
        elif self.strategy == "minimum":
            reward = float(np.min(values))
        elif self.strategy == "confidence_weighted":
            if weights.sum() == 0:
                return EnsembleResult(None, 0, disagreement, variance, entropy, 0, members)
            reward = float(np.average(values, weights=weights))
        coverage = len(available) / len(members)
        confidence = float(np.mean(weights)) * coverage * (1 - disagreement)
        return EnsembleResult(
            reward, confidence, disagreement, variance, entropy, float(np.std(weights)), members
        )
