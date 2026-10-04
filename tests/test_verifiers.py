import pytest

from src.data.schema import Example
from src.rewards.verifiers import ExactVerifier, RuleVerifier, VerifierResult


@pytest.mark.parametrize(
    "reference,response,expected",
    [
        ("42", "Final answer: 42.", 1),
        ("42", "Final answer: 43", 0),
        ("42", "I have no answer", 0),
        ("1200", "Final answer: 1,200", 1),
        ("1/2", r"Final answer: \frac{1}{2}", 1),
    ],
)
def test_exact(reference, response, expected):
    assert ExactVerifier().verify(Example("x", "q", reference), response).reward == expected


@pytest.mark.parametrize(
    "reference,answer,expected",
    [
        ("1/2", "0.5", 1),
        ("-2", "−2.00", 1),
        ("2", "+002", 1),
        ("2/3", "0.6667", 0),
        ("blue", "Blue.", 1),
        ("1", "1/0", 0),
    ],
)
def test_rule(reference, answer, expected):
    assert (
        RuleVerifier().verify(Example("x", "q", reference), f"Final answer: {answer}").reward
        == expected
    )


@pytest.mark.parametrize("reward,confidence", [(2, 1), (-1, 1), (1, -1), (1, 2), (float("nan"), 1)])
def test_invalid_result(reward, confidence):
    with pytest.raises(ValueError):
        VerifierResult(reward, confidence)
