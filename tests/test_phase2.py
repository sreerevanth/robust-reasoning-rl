import json
from pathlib import Path

from src.data.loading import load_dataset
from src.models.generation import RecordedGenerator, policy_prompt, prompt_for
from src.rewards.ensemble import Ensemble
from src.rewards.shaping import RewardConfig
from src.rewards.verifiers import RuleVerifier
from src.training.rewards import RewardFunction


def test_chat_prompt_uses_same_content_without_references():
    question = "How many apples?"
    assert policy_prompt(question, {}) == prompt_for(question)
    assert policy_prompt(question, {"prompt_format": "chat"}) == [
        {"role": "user", "content": prompt_for(question)}
    ]


def test_recorded_generator_replays_measured_responses():
    records = [{"response": "Final answer: 2", "metadata": {"question": "q", "sample_index": 0}}]
    generator = RecordedGenerator(records)
    assert generator.generate("q", 1, 42) == ["Final answer: 2"]
    assert generator.last_generation_metadata[0]["reused_base_generation"]
    import pytest

    with pytest.raises(ValueError):
        generator.generate("q", 2, 42)


def test_deterministic_subset_offsets_and_namespaced_ids(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps({"question": str(i), "reference": "2"}) + "\n" for i in range(10))
    )
    cfg = {"path": str(path), "shuffle_seed": 7, "limit": 4, "id_prefix": "train-"}
    first = load_dataset(cfg)
    assert first == load_dataset(cfg)
    second = load_dataset({**cfg, "offset": 4})
    assert not {e.id for e in first} & {e.id for e in second}


def test_audit_is_logging_only_and_contains_raw_shaped_judgments(tmp_path):
    plain = RewardFunction(Ensemble([RuleVerifier()]), RewardConfig(strategy="combined"))
    audited = RewardFunction(
        Ensemble([RuleVerifier()]),
        RewardConfig(strategy="combined"),
        tmp_path / "audit.jsonl",
        RuleVerifier(),
    )
    args = {
        "completions": ["Final answer: 2"],
        "reference": ["2"],
        "example_id": ["x"],
        "question": ["q"],
        "example_metadata": [{}],
    }
    assert plain(**args) == audited(**args)
    record = json.loads((tmp_path / "audit.jsonl").read_text())
    assert record["observed"]["reward"] == record["shaped"]["value"] == 1
    assert record["independent_correctness"]
    assert audited.totals["observed_reward_square_sum"] == 1


def test_gsm8k_frozen_splits_are_disjoint_and_match_manifest():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "experiments/phase2/dataset_manifest.json").read_text())
    ids, questions = set(), set()
    for split in manifest["splits"].values():
        path = root / split["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == split["sha256"]
        examples = load_dataset({"path": str(path)})
        assert len(examples) == split["count"]
        assert not ids.intersection(e.id for e in examples)
        assert not questions.intersection(e.question for e in examples)
        ids.update(e.id for e in examples)
        questions.update(e.question for e in examples)


def test_archived_evidence_retains_recorded_bytes_without_weights():
    import hashlib

    root = Path(__file__).resolve().parents[1] / "experiments/phase2/artifacts"
    manifests = list(root.glob("*/archive_manifest.json"))
    assert manifests
    for path in manifests:
        manifest = json.loads(path.read_text())
        for entry in manifest["files"]:
            artifact = path.parent / entry["path"]
            assert artifact.stat().st_size == entry["bytes"]
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]
            assert artifact.suffix not in {".bin", ".safetensors"}
