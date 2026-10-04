"""TRL callable accepting aligned prompt/completion dataset columns."""

from collections import Counter
from typing import Any

from src.data.schema import Example
from src.rewards.ensemble import Ensemble
from src.rewards.shaping import RewardConfig, shape_reward
from src.utils.logging import event


class RewardFunction:
    __name__ = "verifier_reward"

    def __init__(self, ensemble: Ensemble, config: RewardConfig):
        self.ensemble, self.config = ensemble, config
        self.statistics: Counter[str] = Counter()
        self.totals = {
            "observed_reward_sum": 0.0,
            "observed_samples": 0.0,
            "shaped_reward_sum": 0.0,
            "disagreement_sum": 0.0,
            "confidence_sum": 0.0,
        }

    def __call__(
        self,
        completions: list,
        reference: list[str],
        example_id: list[str],
        question: list[str],
        example_metadata: list[dict],
        **kwargs: Any,
    ) -> list[float]:
        lengths = {len(x) for x in (completions, reference, example_id, question, example_metadata)}
        if len(lengths) != 1:
            raise ValueError("Reward function columns are misaligned")
        rewards = []
        disagreements = []
        for completion, answer, eid, problem, metadata in zip(
            completions, reference, example_id, question, example_metadata, strict=True
        ):
            response = (
                completion
                if isinstance(completion, str)
                else "".join(message.get("content", "") for message in completion)
            )
            result = self.ensemble.verify(Example(eid, problem, answer, metadata), response)
            shaped = shape_reward(result, self.config)
            rewards.append(shaped.value)
            disagreements.append(result.disagreement)
            if result.reward is not None:
                self.totals["observed_reward_sum"] += result.reward
                self.totals["observed_samples"] += 1
            self.totals["shaped_reward_sum"] += shaped.value
            self.totals["disagreement_sum"] += result.disagreement
            self.totals["confidence_sum"] += result.confidence
            self.statistics.update(
                samples=1,
                suppressed=int(shaped.suppressed),
                downweighted=int(shaped.downweighted),
                missing=int(shaped.missing),
            )
        if rewards:
            event(
                "reward_batch",
                mean_reward=sum(rewards) / len(rewards),
                cumulative_observed_reward=self.totals["observed_reward_sum"]
                / self.totals["observed_samples"]
                if self.totals["observed_samples"]
                else None,
                disagreement=sum(disagreements) / len(disagreements),
                **dict(self.statistics),
            )
        return rewards
