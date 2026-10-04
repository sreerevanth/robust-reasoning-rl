"""Reward attenuation is shaping, not a claim of zero GRPO gradient weight."""

from dataclasses import dataclass

from src.rewards.ensemble import EnsembleResult


@dataclass(frozen=True)
class RewardConfig:
    strategy: str = "standard"
    penalty: float = 0.2
    moderate_threshold: float = 0.4
    severe_threshold: float = 0.9
    moderate_scale: float = 0.5
    severe_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.strategy not in {"standard", "confidence", "penalized", "combined", "conservative"}:
            raise ValueError(f"Unknown reward strategy: {self.strategy}")
        if not 0 <= self.moderate_threshold <= self.severe_threshold <= 1:
            raise ValueError("Require 0 <= moderate <= severe <= 1")
        if not 0 <= self.penalty < float("inf"):
            raise ValueError("penalty must be finite and nonnegative")
        if not 0 <= self.moderate_scale <= 1 or not 0 <= self.severe_scale <= 1:
            raise ValueError("Attenuation scales must be in [0,1]")


@dataclass(frozen=True)
class ShapedReward:
    value: float
    scale: float
    suppressed: bool
    downweighted: bool
    missing: bool


def shape_reward(result: EnsembleResult, config: RewardConfig) -> ShapedReward:
    if result.reward is None:
        return ShapedReward(0, 0, True, False, True)
    reward = result.reward
    if config.strategy == "standard":
        return ShapedReward(reward, 1, False, False, False)
    if config.strategy in {"confidence", "combined"}:
        reward *= result.confidence
    if config.strategy in {"penalized", "combined"}:
        reward -= config.penalty * result.disagreement
    if config.strategy == "conservative":
        reward = min(m.reward for m in result.members if m.reward is not None)
    scale = 1.0
    if result.disagreement >= config.severe_threshold:
        scale = config.severe_scale
    elif result.disagreement >= config.moderate_threshold:
        scale = config.moderate_scale
    return ShapedReward(reward * scale, scale, scale == 0, 0 < scale < 1, False)
