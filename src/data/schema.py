"""Serializable records; references are never passed to the generation backend."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    example_id: str
    response: str
    reasoning: str
    final_answer: str | None
    verifier_outputs: dict[str, Any]
    observed_reward: float | None
    confidence: float
    disagreement: float
    independent_correctness: bool
    metadata: dict[str, Any] = field(default_factory=dict)
