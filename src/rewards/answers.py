"""Bounded parsing of scalar mathematical answers, without eval or execution."""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction


@dataclass(frozen=True)
class ExtractedAnswer:
    reasoning: str
    answer: str | None


def _boxed(text: str) -> tuple[int, str] | None:
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    match = matches[-1]
    depth = 1
    for i in range(match.end(), len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return match.start(), text[match.end():i]
    return match.start(), ""


def extract_answer(response: str) -> ExtractedAnswer:
    """Use the last explicit final marker; only bare scalars get fallback parsing."""
    text = response.strip()
    markers = list(re.finditer(
        r"(?:final\s+answer\s*[:=]|the\s+answer\s+is\s*[:=]?|####)\s*", text, re.I
    ))
    box = _boxed(text)
    if box and (not markers or box[0] > markers[-1].start()):
        answer = box[1].strip()
        return ExtractedAnswer(text[:box[0]].strip(), answer or None)
    if markers:
        match = markers[-1]
        tail = text[match.end():].splitlines()
        answer = tail[0].strip() if tail else ""
        nested = _boxed(answer)
        if nested:
            answer = nested[1]
        answer = answer.strip().strip("$*").rstrip(".!;").strip()
        return ExtractedAnswer(text[:match.start()].strip(), answer or None)
    candidate = text.strip("$ ").rstrip(".!;")
    if numeric_value(candidate) is not None:
        return ExtractedAnswer("", candidate)
    return ExtractedAnswer(text, None)


def normalize_answer(answer: str | None) -> str | None:
    if answer is None or len(answer) > 4096:
        return None
    text = answer.strip().replace("−", "-").replace("＋", "+")
    text = text.strip("$ ").rstrip(".!;")
    text = re.sub(r"\\(?:left|right)", "", text)
    box = _boxed(text)
    if box and box[0] == 0:
        text = box[1]
    text = re.sub(r"\\(?:d?frac|tfrac)\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}", r"\1/\2", text)
    # Strip thousands separators only when grouping is valid, not decimal commas.
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.casefold() or None


def numeric_value(answer: str | None) -> Fraction | None:
    text = normalize_answer(answer)
    if text is None or len(text) > 256:
        return None
    scalar = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?"
    if not re.fullmatch(rf"{scalar}(?:/{scalar})?", text):
        return None
    try:
        parts = text.split("/")
        values = [Fraction(Decimal(part)) for part in parts]
        return values[0] if len(values) == 1 else values[0] / values[1]
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None
