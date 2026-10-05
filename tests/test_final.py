import copy
import hashlib
import json
from pathlib import Path

from src.data.schema import Example
from src.evaluation.runner import evaluate
from src.rewards.answers import extract_answer, numeric_value
from src.rewards.verifiers import RuleVerifier
from src.utils.config import DEFAULTS, load_config


def test_observed_validation_units_remain_wrong_answers():
    for response, answer, reference in [
        ("Final answer: 49,500 televisions", "49,500", "477"),
        ("Final answer: 0.0025 pounds per square inch", "0.0025", "4"),
        (
            r"Final answer: The combined total number of sit-ups performed is \( 470 \)",
            "470",
            "510",
        ),
        (
            "The final answer is: Melissa will groom 5664 animals over the 10-day period.",
            "Melissa will groom 5664 animals over the 10-day period",
            "14",
        ),
    ]:
        assert extract_answer(response).answer == answer
        assert RuleVerifier().verify(Example("x", "q", reference), response).reward == 0


def test_nonscalar_final_answer_counts_as_extraction_failure(tmp_path):
    class ObservedGenerator:
        def generate(self, question, count, seed):
            return ["Final answer: Goldfish = 5 teaspoons"]

    cfg = copy.deepcopy(DEFAULTS)
    cfg["dataset"]["path"] = str(Path(__file__).resolve().parents[1] / "data/tiny_eval.jsonl")
    cfg["output_dir"] = str(tmp_path)
    cfg["generation"]["num_generations"] = 1
    cfg["evaluation"]["k"] = [1]
    result = evaluate(cfg, ObservedGenerator())
    assert result["metrics"]["extraction_failure_rate"] == 1
    assert result["metrics"]["independent_accuracy"] == 0


def test_real_qwen_final_formats_accept_correct_scalars_and_reject_ambiguity():
    for response, reference in [
        ("Final answer: The number is 33", "33"),
        ("Final answer: 12 teaspoons", "12"),
        ("Final answer: <answer>9</answer>", "9"),
        (r"Therefore, the total amount of money Erick collected is **\$2220**.", "2220"),
    ]:
        assert RuleVerifier().verify(Example("x", "q", reference), response).reward == 1
    for response in [
        "Final answer: not 9",
        "Final answer: 9 or 10",
        "Final answer: 2 + 7 = 9",
        "I used 9 in the calculation",
        "Therefore, consider 9 before computing 10.",
    ]:
        assert RuleVerifier().verify(Example("x", "q", "9"), response).reward == 0


def test_final_splits_hashes_and_qualification_separation():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "experiments/final/dataset_manifest.json").read_text())
    validation_path = root / manifest["validation"]["path"]
    assert (
        hashlib.sha256(validation_path.read_bytes()).hexdigest() == manifest["validation"]["sha256"]
    )
    pools = {}
    for split, item in manifest["splits"].items():
        path = root / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == item["count"]
        assert all(numeric_value(row["reference"]) is not None for row in rows)
        pools[split] = {row["question"] for row in rows}
    validation = {
        json.loads(line)["question"]
        for line in (root / "data/gsm8k_phase2_validation.jsonl").read_text().splitlines()
    }
    inspected = {
        json.loads(line)["question"]
        for line in (root / "data/gsm8k_phase2_test.jsonl").read_text().splitlines()
    }
    assert not pools["train"] & pools["test"]
    assert not pools["train"] & validation
    assert not pools["test"] & inspected


def test_nested_protocol_paths_resolve_from_repository(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    protocol = tmp_path / "experiments/final/protocol.yaml"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("dataset: {source: local, path: data/test.jsonl}\n")
    config = load_config(protocol)
    assert config["dataset"]["path"] == str(tmp_path / "data/test.jsonl")
