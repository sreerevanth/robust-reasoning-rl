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
    assert len(plots) == 9 and all(path.stat().st_size > 1000 for path in plots)


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
