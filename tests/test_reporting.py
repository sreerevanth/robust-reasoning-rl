import json

import pytest

from src.experiments.reporting import create_report


def test_report_aggregates_sample_std_and_preserves_missing(tmp_path):
    rows = []
    for seed, value in [(42, 0.0), (123, 0.5), (456, 1.0)]:
        rows.append(
            {
                "seed": seed,
                "method": "robust",
                "corruption_level": 0.2,
                "pass@1": value,
                "pass@4": value,
                "independent_accuracy": value,
                "observed_reward": value,
                "reward_hacking_gap": 0.0,
                "false_positive_reward_rate": None,
                "false_negative_rate": None,
                "verifier_disagreement": 0.0,
                "extraction_failure_rate": 0.0,
            }
        )
    (tmp_path / "summary.json").write_text(json.dumps({"runs": rows, "failures": []}))
    report = create_report(tmp_path)
    assert report["runs"] == 3 and report["aggregate_rows"] == 1
    import pandas as pd

    frame = pd.read_csv(tmp_path / "aggregate.csv")
    assert frame["independent_accuracy_mean"][0] == 0.5
    assert frame["independent_accuracy_std"][0] == 0.5
    assert "undefined" in (tmp_path / "report.md").read_text()
    assert "No qualifying measured example" in (tmp_path / "failure_analysis.md").read_text()
    (tmp_path / "summary.json").write_text(json.dumps({"runs": rows + [rows[0]], "failures": []}))
    with pytest.raises(ValueError, match="Duplicate"):
        create_report(tmp_path)
