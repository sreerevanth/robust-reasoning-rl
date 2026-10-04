import pytest

from src.rewards.answers import extract_answer, normalize_answer, numeric_value


@pytest.mark.parametrize(
    "text,answer",
    [
        ("The answer is 42.", "42"),
        ("Final answer: -3.50", "-3.50"),
        (r"work \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        ("#### 1,200", "1,200"),
        ("Final answer: **42**.", "42"),
        ("  +4  ", "+4"),
        ("Final answer: 2\nmore work", "2"),
        ("No final answer", None),
        (r"work \boxed{42", None),
        ("Final answer:", None),
        ("Final answer: 2\nFinal answer: 3", "3"),
    ],
)
def test_extraction(text, answer):
    assert extract_answer(text).answer == answer


@pytest.mark.parametrize(
    "a,b",
    [
        ("+0042", "42.0"),
        ("1,200", "1200"),
        ("−2", "-2"),
        (r"\frac{1}{2}", ".5"),
        ("3/6", "0.50"),
        ("1e2", "100"),
    ],
)
def test_numeric_equivalence(a, b):
    assert numeric_value(a) == numeric_value(b) is not None


@pytest.mark.parametrize("text", ["1/0", "nan", "inf", "__import__('os')", "1,2", "x+1"])
def test_reject_unsafe_or_invalid_numbers(text):
    assert numeric_value(text) is None


def test_normalization_and_separation():
    assert normalize_answer("  HELLO world. ") == "helloworld"
    parsed = extract_answer("Compute 6*7. Final answer: 42.")
    assert parsed.reasoning == "Compute 6*7."
    assert normalize_answer(None) is None
