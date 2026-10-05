"""Archive actual lightweight evidence and plots, excluding model weights/checkpoints."""

import argparse
import hashlib
import shutil
from pathlib import Path

from src.utils.persistence import write_json

NAMES = {
    "summary.csv",
    "summary.json",
    "manifest.json",
    "evaluation.json",
    "generations.jsonl",
    "training.json",
    "run.json",
    "reward_audit.jsonl",
    "failure.json",
    "report.md",
    "aggregate.csv",
    "per_seed.csv",
    "failure_analysis.json",
    "failure_analysis.md",
    "training_diagnostics.csv",
    "paired_method_differences.csv",
    "analysis.json",
    "execution_status.json",
    "gate.json",
}


def archive(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Evidence destination must be outside the live result directory")
    entries = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or any(
            p == "final" or p.startswith("checkpoint-") for p in relative.parts
        ):
            continue
        if path.name not in NAMES and not (
            "plots" in relative.parts and path.suffix in {".png", ".pdf"}
        ):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    write_json(
        destination / "archive_manifest.json", {"source_root": str(source), "files": entries}
    )
    print(f"Archived {len(entries)} measured evidence files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    archive(Path(args.results), Path(args.destination))
