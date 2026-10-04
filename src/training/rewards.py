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
                disagreement=sum(disagreements) / len(disagreements),
                **dict(self.statistics),
            )
        return rewards
