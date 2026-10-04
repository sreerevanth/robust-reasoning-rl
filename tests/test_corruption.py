import pytest

from src.data.schema import Example
from src.rewards.corruption import CorruptedVerifier, CorruptionConfig
from src.rewards.verifiers import RuleVerifier


@pytest.mark.parametrize(
    "kind,answer,expected",
    [
        ("flip", "2", 0),
        ("flip", "3", 1),
        ("false_positive", "3", 1),
        ("false_positive", "2", 1),
        ("false_negative", "2", 0),
        ("false_negative", "3", 0),
        ("missing", "2", None),
        ("bias", "3", 1),
    ],
)
def test_corruption_modes(kind, answer, expected):
    verifier = CorruptedVerifier(RuleVerifier(), CorruptionConfig(kind=kind, probability=1))
    result = verifier.verify(
        Example("x", "q", "2", {"class": "addition"}), f"Final answer: {answer}"
    )
    assert result.reward == expected
    if expected is None:
        assert result.confidence == 0


def test_seeded_order_independent_and_confidence():
    cfg = CorruptionConfig(probability=0.4, seed=8, confidence_scale=0.2)
    first, second = CorruptedVerifier(RuleVerifier(), cfg), CorruptedVerifier(RuleVerifier(), cfg)
    examples = [Example(str(i), "q", "2") for i in range(100)]
    a = {e.id: first.verify(e, "Final answer: 2") for e in examples}
    b = {e.id: second.verify(e, "Final answer: 2") for e in reversed(examples)}
    assert a == b
    assert {r.reward for r in a.values()} == {0, 1}
    assert all(r.confidence == 0.2 for r in a.values())
    other = CorruptedVerifier(RuleVerifier(), CorruptionConfig(probability=0.4, seed=9))
    assert any(a[e.id].reward != other.verify(e, "Final answer: 2").reward for e in examples)


def test_noise_and_bias_subset():
    noisy = CorruptedVerifier(
        RuleVerifier(), CorruptionConfig(kind="noise", probability=1, noise_std=0.5)
    )
    values = [noisy.verify(Example(str(i), "q", "2"), "Final answer: 2").reward for i in range(50)]
    assert all(0 <= v <= 1 for v in values)
    assert any(0 < v < 1 for v in values)
    biased = CorruptedVerifier(RuleVerifier(), CorruptionConfig(kind="bias", probability=1))
    assert (
        biased.verify(Example("x", "q", "2", {"class": "division"}), "Final answer: 3").reward == 0
    )


@pytest.mark.parametrize(
    "kwargs", [{"probability": -1}, {"confidence_scale": 2}, {"kind": "bad"}, {"noise_std": -1}]
)
def test_invalid_corruption(kwargs):
    with pytest.raises(ValueError):
        CorruptionConfig(**kwargs)
