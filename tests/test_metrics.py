import math

import pytest

from src.data.schema import Generation
from src.evaluation.metrics import compute_metrics, pass_at_k


@pytest.mark.parametrize(
    "n,c,k,expected",
    [
        (4, 0, 1, 0),
        (4, 4, 4, 1),
        (4, 1, 2, 0.5),
        (10, 2, 3, 1 - math.comb(8, 3) / math.comb(10, 3)),
    ],
)
def test_pass_at_k(n, c, k, expected):
    assert pass_at_k(n, c, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    "args", [(0, 0, 1), (4, 5, 1), (4, 1, 5), (4, -1, 1), (4, 1, 0), (4.0, 1, 1)]
)
def test_invalid_pass(args):
    with pytest.raises(ValueError):
        pass_at_k(*args)


def record(correct, reward):
    return Generation("x", "text", "work", "1" if correct else "2", {}, reward, 1, 0, correct)


def test_hacking_and_missingness():
    result = compute_metrics([record(False, 1), record(True, 0), record(False, None)], [1, 2])
    assert result["false_positive_reward_rate"] == 1
    assert result["false_negative_rate"] == 1
    assert result["reward_correctness_correlation"] == pytest.approx(-1)
    assert result["reward_coverage"] == pytest.approx(2 / 3)
    assert result["reward_hacking_gap"] == 0
    assert result["pass@1"] == pytest.approx(1 / 3)
    assert result["pass@2"] == pytest.approx(2 / 3)
    assert result["answer_consistency"] == pytest.approx(1 / 3)


def test_no_rewards_and_constant_correlation():
    result = compute_metrics([record(True, None)], [1])
    assert result["observed_reward"] is None
    assert result["reward_hacking_gap"] is None
    assert result["answer_consistency"] is None
    assert compute_metrics([record(True, 1)], [1])["reward_correctness_correlation"] is None
    with pytest.raises(ValueError):
        compute_metrics([], [1])
