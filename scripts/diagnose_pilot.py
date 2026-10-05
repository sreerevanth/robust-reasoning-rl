"""Produce a bounded audit of the original CPU pilot without rewriting its results."""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.schema import Example
from src.rewards.answers import numeric_value
from src.rewards.verifiers import RuleVerifier
from src.utils.persistence import write_json


def diagnose(root: Path, destination: Path) -> None:
    records = [
        json.loads(line)
        for path in sorted(root.glob("seed-*/level-0/base/evaluation/generations.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not records:
        raise ValueError("No original uncorrupted base-model records")
    examples = {
        r["example_id"]: Example(
            r["example_id"], r["metadata"]["question"], r["metadata"]["reference"]
        )
        for r in records
    }
    assert all(numeric_value(e.reference) is not None for e in examples.values())
    assert all(
        RuleVerifier().verify(e, "Final answer: " + e.reference).reward == 1
        for e in examples.values()
    )
    counts = Counter()
    cases = {}
    for record in records:
        status = (
            "truncated"
            if record["metadata"]["finish_reason"] == "length"
            else "missing_marker"
            if record["final_answer"] is None
            else "nonscalar_final"
            if numeric_value(record["final_answer"]) is None
            else "parsed_wrong"
            if not record["independent_correctness"]
            else "correct"
        )
        counts[status] += 1
        cases.setdefault(
            status,
            {
                "example_id": record["example_id"],
                "reference": record["metadata"]["reference"],
                "response_excerpt": record["response"][-800:],
            },
        )
    write_json(
        destination,
        {
            "source": str(root),
            "records": len(records),
            "categories": dict(counts),
            "cases": cases,
            "gold_acceptance_checks": len(examples),
            "interpretation": "Categories are ordered: truncation takes precedence."
            " Wrong visible numbers are not rescued by arbitrary-number extraction.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    diagnose(args.results, args.output)
