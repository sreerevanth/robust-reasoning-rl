import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest

from src.data.loading import load_dataset
from src.utils.config import DEFAULTS, load_config, validate
from src.utils.persistence import provenance, write_json
from src.utils.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["baseline", "robust", "evaluation", "reward_corruption", "smoke"])
def test_config_and_dataset(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    examples = load_dataset(cfg["dataset"])
    assert len(examples) == 4
    assert len({e.id for e in examples}) == 4


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("generation", "num_generations", 0),
        ("generation", "temperature", -1),
        ("generation", "top_p", 2),
        ("evaluation", "k", [5]),
        ("training", "beta", -1),
        ("reward", "strategy", "bad"),
    ],
)
def test_invalid_config(section, key, value):
    cfg = copy.deepcopy(DEFAULTS)
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate(cfg)


def test_seed_and_serialization(tmp_path):
    seed_everything(12)
    first = (random.random(), np.random.rand())
    seed_everything(12)
    assert first == (random.random(), np.random.rand())
    value = provenance(DEFAULTS, method="test")
    write_json(tmp_path / "nested" / "run.json", value)
    assert json.loads((tmp_path / "nested" / "run.json").read_text()) == value
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"a": float("nan")})
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_local_data_and_independent_corruption(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id":"a","question":"q","reference":"1"}\n' * 2)
    with pytest.raises(ValueError):
        load_dataset({"path": str(path)})
    cfg = copy.deepcopy(DEFAULTS)
    cfg["evaluation_verifier"]["corruption"] = {"kind": "flip"}
    with pytest.raises(ValueError):
        validate(cfg)


@pytest.mark.parametrize("permanent", [False, True])
def test_atomic_write_handles_windows_lock_without_losing_old_file(
    tmp_path, monkeypatch, permanent
):
    import os

    target = tmp_path / "run.json"
    target.write_text('{"old": true}')
    replace = os.replace
    calls = []

    def locked_replace(source, destination):
        calls.append(source)
        if permanent or len(calls) == 1:
            raise PermissionError("simulated Windows sync lock")
        replace(source, destination)

    monkeypatch.setattr("src.utils.persistence.os.replace", locked_replace)
    monkeypatch.setattr("src.utils.persistence.time.sleep", lambda delay: None)
    if permanent:
        with pytest.raises(PermissionError):
            write_json(target, {"new": True})
        assert json.loads(target.read_text()) == {"old": True}
        assert len(calls) == 5
    else:
        write_json(target, {"new": True})
        assert json.loads(target.read_text()) == {"new": True}
        assert len(calls) == 2
    assert not list(tmp_path.glob("*.tmp"))
