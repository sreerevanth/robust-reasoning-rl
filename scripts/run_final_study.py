"""One-command CUDA qualification, resumable sweep, reporting, plots, and evidence export."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from src.evaluation.runner import evaluate
from src.experiments.reporting import create_report
from src.experiments.runner import experiment
from src.utils.config import load_config
from src.utils.persistence import provenance, write_json
from src.visualization.plots import plot_results

ROOT = Path(__file__).resolve().parents[1]


def run(protocol: Path, resume: bool = False, check_only: bool = False) -> int:
    import torch

    config = load_config(protocol)
    output = Path(config["output_dir"])
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    hardware = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "protocol_sha256": protocol_hash,
    }
    if not torch.cuda.is_available():
        write_json(
            output / "execution_status.json",
            provenance(
                config,
                status="externally_blocked",
                reason="CUDA unavailable",
                hardware=hardware,
            ),
        )
        print("CUDA unavailable: final study not executed. CPU diagnostics are separate evidence.")
        return 2
    hardware.update(
        gpu=torch.cuda.get_device_name(),
        vram_bytes=torch.cuda.get_device_properties(0).total_memory,
    )
    if check_only:
        print(json.dumps(hardware, indent=2))
        return 0
    qualification_path = output / "qualification" / "evaluation.json"
    gate = output / "qualification" / "gate.json"
    if resume and gate.exists() and qualification_path.exists():
        saved_gate = json.loads(gate.read_text())
        if saved_gate["protocol_sha256"] != protocol_hash:
            raise ValueError("Qualification protocol changed; use a new output directory")
        qualification = json.loads(qualification_path.read_text())
        if (
            hashlib.sha256(qualification_path.read_bytes()).hexdigest()
            != saved_gate["evidence_sha256"]
        ):
            raise ValueError("Qualification evidence changed")
    else:
        qualification_config = copy.deepcopy(config)
        qualification_config["dataset"] = config["experiment"]["qualification_dataset"]
        qualification_config["output_dir"] = str(qualification_path.parent)
        qualification = evaluate(qualification_config)
        write_json(
            gate,
            {
                "protocol_sha256": protocol_hash,
                "evidence_sha256": hashlib.sha256(qualification_path.read_bytes()).hexdigest(),
            },
        )
    if qualification["metrics"]["pass@1"] < config["experiment"]["qualification_min_pass1"]:
        write_json(
            output / "execution_status.json",
            provenance(
                config,
                status="qualification_failed",
                hardware=hardware,
                qualification_metrics=qualification["metrics"],
            ),
        )
        print("Capability gate failed. No RL sweep launched; qualification evidence retained.")
        return 3
    config["experiment"]["resume"] = resume
    result = experiment(config)
    if result["runs"]:
        create_report(output)
        plot_results(output)
        evidence = output.parent / (output.name + "-evidence")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/archive_results.py"),
                "--results",
                str(output),
                "--destination",
                str(evidence),
            ],
            check=True,
        )
        with zipfile.ZipFile(evidence.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in evidence.rglob("*"):
                if path.is_file():
                    bundle.write(path, path.relative_to(evidence))
    write_json(
        output / "execution_status.json",
        provenance(
            config,
            status="completed" if not result["failures"] else "incomplete",
            hardware=hardware,
            completed=len(result["runs"]),
            failed=len(result["failures"]),
        ),
    )
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/final/protocol.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.resume, args.check_only))
