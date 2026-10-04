"""Deterministic answer verifiers with a shared extensible protocol."""

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.data.schema import Example
from src.rewards.answers import extract_answer, normalize_answer, numeric_value


@dataclass(frozen=True)
class VerifierResult:
    reward: float | None
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be finite in [0,1]")
        if self.reward is not None and (not math.isfinite(self.reward) or not 0 <= self.reward <= 1):
            raise ValueError("Reward must be None or finite in [0,1]")


class Verifier(Protocol):
    def verify(self, example: Example, response: str) -> VerifierResult: ...


class ExactVerifier:
    """Text comparison after conservative formatting normalization."""

    def verify(self, example: Example, response: str) -> VerifierResult:
        answer = extract_answer(response).answer
        normalized = normalize_answer(answer)
        correct = normalized is not None and normalized == normalize_answer(example.reference)
        return VerifierResult(float(correct), 1.0, {"answer": answer, "kind": "exact"})


class RuleVerifier:
    """Exact rational scalar equivalence, with normalized text fallback."""

    def verify(self, example: Example, response: str) -> VerifierResult:
        answer = extract_answer(response).answer
        actual, reference = numeric_value(answer), numeric_value(example.reference)
        if actual is not None and reference is not None:
            return VerifierResult(float(actual == reference), 1.0, {"kind": "rational"})
        return ExactVerifier().verify(example, response)
