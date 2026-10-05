import json
from pathlib import Path

import pytest

from src.cli import main
from src.evaluation.runner import evaluate
from src.experiments.runner import experiment
from src.utils.config import load_config
from src.visualization.plots import plot_results

ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_reproducible_and_cli(tmp_path):
    cfg = load_config(ROOT / "configs/evaluation.yaml")
    cfg["output_dir"] = str(tmp_path / "eval")
    first, second = evaluate(cfg), evaluate(cfg)
    assert first["records"] == second["records"]
    assert first["metrics"] == second["metrics"]
    assert first["metadata"]["execution_kind"] == "software_fixture"
    lines = (tmp_path / "eval/generations.jsonl").read_text().splitlines()
    assert len(lines) == 16 and all(json.loads(line) for line in lines)
    assert main(["validate", "--config", str(ROOT / "configs/baseline.yaml")]) == 0
    assert main(["evaluate", "--config", str(tmp_path / "missing.yaml")]) == 1


def test_fixture_sweep_and_plots(tmp_path):
    cfg = load_config(ROOT / "configs/smoke.yaml")
    cfg["output_dir"] = str(tmp_path)
    cfg["experiment"]["seeds"] = [42]
    result = experiment(cfg)
    assert len(result["runs"]) == 5
    assert not result["failures"]
    assert {r["method"] for r in result["runs"]} == {"fixture_verifier_audit"}
    assert len({r["independent_accuracy"] for r in result["runs"]}) == 1
    plots = plot_results(tmp_path)
    assert len(plots) == 13 and all(path.stat().st_size > 1000 for path in plots)


def test_resume_keeps_completed_evidence_and_rejects_changed_protocol(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs/smoke.yaml")
    cfg["output_dir"] = str(tmp_path)
    cfg["experiment"].update(seeds=[42], levels=[0, 0.2])
    first = experiment(cfg)
    cfg["experiment"]["resume"] = True

    def fail_if_called(config):
        raise AssertionError("Completed condition was rerun")

    monkeypatch.setattr("src.experiments.runner.evaluate", fail_if_called)
    assert experiment(cfg)["runs"] == first["runs"]
    evidence = tmp_path / "seed-42/level-0/fixture_verifier_audit/evaluation/evaluation.json"
    original = evidence.read_bytes()
    evidence.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="evidence changed"):
        experiment(cfg)
    evidence.write_bytes(original)
    cfg["generation"]["max_tokens"] += 1
    with pytest.raises(ValueError, match="changed configuration"):
        experiment(cfg)


def test_sweep_rejects_fake_training_and_records_failure(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs/smoke.yaml")
    cfg["output_dir"] = str(tmp_path)
    cfg["experiment"]["mode"] = "train"
    with pytest.raises(ValueError):
        experiment(cfg)
    cfg["experiment"]["mode"] = "fixture"

    def fail(config):
        raise RuntimeError("controlled test failure")

    monkeypatch.setattr("src.experiments.runner.evaluate", fail)
    with pytest.raises(RuntimeError):
        experiment(cfg)
    assert json.loads((tmp_path / "summary.json").read_text())["failures"]


def test_fair_baseline_robust_configs():
    baseline = load_config(ROOT / "configs/baseline.yaml")
    robust = load_config(ROOT / "configs/robust.yaml")
    for key in ("model", "dataset", "generation", "training", "training_verifier", "seed"):
        assert baseline[key] == robust[key]
    assert baseline["reward"] != robust["reward"]


def test_resume_retries_failed_condition_and_retains_failure_history(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs/smoke.yaml")
    cfg["output_dir"] = str(tmp_path)
    cfg["experiment"].update(seeds=[42], levels=[0, 0.2], continue_on_error=True)

    def fail_noisy_condition(config):
        if config["training_verifier"]["members"][0]["corruption"]["probability"] == 0.2:
            raise RuntimeError("controlled interruption")
        return evaluate(config)

    monkeypatch.setattr("src.experiments.runner.evaluate", fail_noisy_condition)
    first = experiment(cfg)
    assert len(first["runs"]) == len(first["failures"]) == 1
    assert first["failures"][0]["config"]["model"] == cfg["model"]
    monkeypatch.setattr("src.experiments.runner.evaluate", evaluate)
    cfg["experiment"]["resume"] = True
    resumed = experiment(cfg)
    assert len(resumed["runs"]) == 2 and not resumed["failures"]
    assert len(resumed["failure_history"]) == 1
    assert resumed["runs"][0] == first["runs"][0]


def test_real_sweep_orchestration_preserves_controlled_comparison(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs/reward_corruption.yaml")
    cfg["output_dir"] = str(tmp_path)
    cfg["experiment"].update(seeds=[42], levels=[0.2])
    training_configs = []
    evaluation_configs = []

    def record_training(config):
        training_configs.append(config)
        return {
            "checkpoint": str(Path(config["output_dir"]) / "final"),
            "metrics": {"train_loss": 0.25},
            "reward_statistics": {"samples": 4},
        }

    def record_evaluation(config):
        evaluation_configs.append(config)
        return {"metrics": {"independent_accuracy": 0.5}}

    monkeypatch.setattr("src.experiments.runner.train", record_training)
    monkeypatch.setattr("src.experiments.runner.evaluate", record_evaluation)
    result = experiment(cfg)
    assert [row["method"] for row in result["runs"]] == ["base", "baseline", "robust"]
    assert len(training_configs) == 2
    baseline, robust = training_configs
    for section in ("model", "dataset", "generation", "training", "training_verifier", "seed"):
        assert baseline[section] == robust[section]
    assert baseline["reward"]["strategy"] == "standard"
    assert robust["reward"]["strategy"] == "combined"
    assert "adapter" not in evaluation_configs[0]["model"]
    assert all("adapter" in run["model"] for run in evaluation_configs[1:])


def test_train_eval_leakage_rejected(tmp_path):
    cfg = load_config(ROOT / "configs/reward_corruption.yaml")
    cfg["output_dir"] = str(tmp_path)
    cfg["experiment"]["train_dataset"] = cfg["dataset"]
    with pytest.raises(ValueError, match="overlap"):
        experiment(cfg)
