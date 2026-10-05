"""Audit actual candidate outputs without changing their original evaluation artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.runner import evaluate
from src.experiments.reporting import markdown_table
from src.models.generation import RecordedGenerator
from src.utils.persistence import write_json


def report(roots: list[Path], destination: Path) -> None:
    rows = []
    for root in roots:
        original = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
        config = original["metadata"]["config"]
        config["output_dir"] = str(root / "parser-audit")
        audited = evaluate(config, RecordedGenerator(original["records"]))
        rows.append(
            {
                "model": config["model"]["name"],
                "advertised_parameter_scale": {
                    "HuggingFaceTB/SmolLM2-135M-Instruct": "135M",
                    "Qwen/Qwen2.5-0.5B-Instruct": "0.5B",
                }.get(config["model"]["name"], "unspecified"),
                "model_revision": config["model"]["revision"],
                "examples": original["metrics"]["num_examples"],
                "pass@1": audited["metrics"]["pass@1"],
                "original_pass@1": original["metrics"]["pass@1"],
                "extraction_failure_rate": audited["metrics"]["extraction_failure_rate"],
                "original_extraction_failure_rate": original["metrics"]["extraction_failure_rate"],
                "truncation_rate": original["metrics"]["truncation_rate"],
                "seconds": original["metadata"]["elapsed_seconds"],
                "device": config["model"]["device"],
                "max_tokens": config["generation"]["max_tokens"],
                "temperature": config["generation"]["temperature"],
                "qualification_gate_passed": audited["metrics"]["pass@1"] >= 0.1,
            }
        )
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "candidates.csv", index=False)
    write_json(destination / "candidates.json", rows)
    (destination / "report.md").write_text(
        "# Measured model qualification\n\n"
        "Fixed 16-question validation pool; no final-test selection. "
        "Greedy decoding, one attempt.\n"
        "This is model selection evidence, not an RL result or a statistically strong benchmark.\n"
        "Original outputs are retained. Parser-audit metrics rejudge those same actual outputs\n"
        "with the corrected scalar extraction-failure definition; "
        "no new generations are implied.\n\n" + markdown_table(frame) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    report(args.results, args.destination)
