import pytest

from src.data.schema import Example
from src.rewards.ensemble import Ensemble
from src.rewards.shaping import RewardConfig, shape_reward
from src.rewards.verifiers import VerifierResult
from src.training.rewards import RewardFunction


class Constant:
    def __init__(self, reward, confidence=1):
        self.result = VerifierResult(reward, confidence)

    def verify(self, example, response):
        return self.result


@pytest.mark.parametrize(
    "strategy,expected",
    [("mean", 0.5), ("majority", 0.5), ("minimum", 0), ("confidence_weighted", 0.8)],
)
def test_ensemble(strategy, expected):
    result = Ensemble([Constant(0, 0.2), Constant(1, 0.8)], strategy).verify(
        Example("x", "q", "1"), "text"
    )
    assert result.reward == pytest.approx(expected)
    assert result.disagreement == 1
    assert result.variance == 0.25
    assert result.vote_entropy == 1
    assert result.confidence_dispersion == pytest.approx(0.3)
    assert result.confidence == 0
    assert len(result.to_dict()["members"]) == 2


def test_missing_and_coverage():
    e = Example("x", "q", "1")
    result = Ensemble([Constant(None, 0), Constant(1)]).verify(e, "text")
    assert result.reward == 1
    assert result.confidence == 0.5
    missing = Ensemble([Constant(None, 0)]).verify(e, "text")
    assert missing.reward is None
    assert shape_reward(missing, RewardConfig()).missing
    assert Ensemble([Constant(1, 0)], "confidence_weighted").verify(e, "text").reward is None


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("standard", 0.75),
        ("confidence", 0.1875),
        ("penalized", 0.15),
        ("combined", 0.0875),
        ("conservative", 1 / 6),
    ],
)
def test_reward_strategies(strategy, expected):
    result = Ensemble([Constant(1), Constant(0.5)]).verify(Example("x", "q", "1"), "text")
    # Variance=.0625; disagreement=.25; confidence=.75.
    cfg = RewardConfig(
        strategy=strategy,
        penalty=1.2,
        moderate_threshold=0.2,
        severe_threshold=0.9,
        moderate_scale=1 / 3,
    )
    assert shape_reward(result, cfg).value == pytest.approx(expected)


def test_severe_suppression_and_training_callback():
    ensemble = Ensemble([Constant(1), Constant(0)])
    result = shape_reward(
        ensemble.verify(Example("x", "q", "1"), "text"), RewardConfig(strategy="combined")
    )
    assert result.suppressed and result.value == 0
    reward = RewardFunction(ensemble, RewardConfig(strategy="combined"))
    values = reward(
        completions=[[{"content": "Final answer: 1"}]],
        reference=["1"],
        example_id=["x"],
        question=["q"],
        example_metadata=[{}],
    )
    assert values == [0]
    assert reward.statistics["suppressed"] == 1
    with pytest.raises(ValueError):
        reward(completions=[], reference=["1"], example_id=[], question=[], example_metadata=[])
