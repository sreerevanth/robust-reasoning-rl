"""Generate tables and balanced qualitative examples from measured run artifacts."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.persistence import write_json


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return "undefined"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def create_report(results: str | Path) -> dict[str, Any]:
    root = Path(results)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(summary["runs"])
    if frame.empty:
        raise ValueError("No completed measured runs to report")
    if frame.duplicated(["seed", "corruption_level", "method"]).any():
        raise ValueError("Duplicate run keys would bias seed aggregation")
    frame = frame.sort_values(["corruption_level", "method", "seed"])
    columns = ["corruption_level", "method", "seed", "pass@1"]
    columns += sorted([str(c) for c in frame if str(c).startswith("pass@") and c != "pass@1"])
    columns += [
        "independent_accuracy",
        "observed_reward",
        "reward_hacking_gap",
        "false_positive_reward_rate",
        "false_negative_rate",
        "verifier_disagreement",
        "extraction_failure_rate",
    ]
    measured = frame[columns]
    aggregate_rows = []
    metrics = columns[3:]
    metrics += [
        c
        for c in (
            "training_observed_reward",
            "training_shaped_reward",
            "training_independent_accuracy",
            "training_reward_variance",
            "truncation_rate",
            "mean_generated_tokens",
        )
        if c in frame
    ]
    for (level, method), subset in frame.groupby(["corruption_level", "method"], sort=True):
        row: dict[str, Any] = {
            "corruption_level": level,
            "method": method,
            "seed_count": subset["seed"].nunique(),
        }
        for metric in metrics:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    paired_rows = []
    by_key = {(r["seed"], r["corruption_level"], r["method"]): r for r in summary["runs"]}
    for (seed, level, method), robust in by_key.items():
        baseline = by_key.get((seed, level, "baseline"))
        if method == "robust" and baseline is not None:
            row = {"seed": seed, "corruption_level": level}
            for metric in (
                "independent_accuracy",
                "pass@1",
                "pass@4",
                "observed_reward",
                "reward_hacking_gap",
            ):
                if robust.get(metric) is not None and baseline.get(metric) is not None:
                    row[f"robust_minus_baseline_{metric}"] = robust[metric] - baseline[metric]
            paired_rows.append(row)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(root / "paired_method_differences.csv", index=False)
    config = summary["metadata"]["config"]
    planned = config.get("experiment", {})
    expected_methods = (
        ["fixture_verifier_audit"]
        if planned.get("mode") == "fixture"
        else ["base", "baseline", "robust"]
    )
    missing = [
        {"seed": seed, "level": level, "method": method}
        for seed in planned.get("seeds", [config["seed"]])
        for level in planned.get("levels", [])
        for method in expected_methods
        if (seed, level, method) not in by_key
    ]
    write_json(
        root / "analysis.json",
        {
            "completed_conditions": len(frame),
            "recorded_failures": summary["failures"],
            "uncompleted_planned_conditions": missing,
            "all_independent_correctness_zero": bool((frame["independent_accuracy"] == 0).all()),
            "unique_questions_per_evaluation": sorted(
                frame.get("num_examples", pd.Series(dtype=int)).unique().tolist()
            ),
            "paired_differences": paired_rows,
            "interpretation": (
                "Descriptive preliminary evidence. Synthetic reward inflation "
                "is not proof of learned exploitation; "
                "no significance or capability superiority claim."
            ),
        },
    )
    measured.to_csv(root / "per_seed.csv", index=False)
    aggregate.to_csv(root / "aggregate.csv", index=False)
    compact = aggregate[
        [
            "corruption_level",
            "method",
            "seed_count",
            "independent_accuracy_mean",
            "independent_accuracy_std",
            "reward_hacking_gap_mean",
            "reward_hacking_gap_std",
        ]
    ]
    text = (
        "# Measured experiment report\n\nAutomatically generated from summary.json. "
        "No experimental values are manually supplied.\n\n"
        f"Completed conditions: {len(frame)}. Recorded failures: {len(summary['failures'])}.\n\n"
        "## Per seed\n\n"
        + markdown_table(measured)
        + "\n\n## Across seeds\n\n"
        + markdown_table(compact)
        + "\n\n"
        "Standard deviations are sample standard deviations across seeds. Undefined rates "
        "stay undefined. Tiny budgets do not support significance testing or causal claims. "
        "Repeated base responses across corruption levels are not independent evidence.\n"
    )
    if missing:
        text += (
            f"\n**Incomplete campaign:** {len(missing)} planned conditions remain uncompleted.\n"
        )
    if (frame["independent_accuracy"] == 0).all():
        text += (
            "\nEvery completed condition has zero independently judged correctness. "
            "This correctness floor prevents a conclusion about comparative reasoning gains. "
            "Reward gaps can reflect injected errors rather than learned exploitation.\n"
        )
    (root / "report.md").write_text(text, encoding="utf-8", newline="\n")
    diagnostics = []
    for path in sorted(root.glob("seed-*/level-*/*/training/training.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload["trainer_log_history"]
        cfg = payload["metadata"]["config"]
        row = {
            "seed": cfg["seed"],
            "method": path.parents[1].name,
            "corruption_level": cfg["training_verifier"]["members"][0]
            .get("corruption", {})
            .get("probability"),
            "checkpoint": payload["checkpoint"],
            **payload["metrics"],
            **payload["reward_statistics"],
        }
        audit = path.parent / "reward_audit.jsonl"
        if audit.exists():
            audit_rows = [
                json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines() if line
            ]
            row["training_unique_examples"] = len({r["example_id"] for r in audit_rows})
        for metric in (
            "grad_norm",
            "kl",
            "frac_reward_zero_std",
            "entropy",
            "completions/clipped_ratio",
        ):
            logged_values = [event[metric] for event in history if metric in event]
            row[f"{metric}_mean"] = (
                sum(logged_values) / len(logged_values) if logged_values else None
            )
        checkpoint = Path(payload["checkpoint"]) / "adapter_model.safetensors"
        if checkpoint.exists():
            from safetensors.torch import load_file

            tensors = load_file(str(checkpoint))
            # PEFT default initializes LoRA B to zero; nonzero B confirms an actual adapter update.
            row["lora_b_norm"] = (
                sum(
                    float(value.float().square().sum())
                    for key, value in tensors.items()
                    if "lora_B" in key
                )
                ** 0.5
            )
        diagnostics.append(row)
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(root / "training_diagnostics.csv", index=False)
    categories = {
        "answer_correct_reward_positive": lambda r: (
            r["independent_correctness"]
            and r["observed_reward"] is not None
            and r["observed_reward"] >= 0.5
        ),
        "answer_incorrect_reward_negative": lambda r: (
            not r["independent_correctness"]
            and r["observed_reward"] is not None
            and r["observed_reward"] < 0.5
        ),
        "answer_incorrect_false_positive": lambda r: (
            not r["independent_correctness"]
            and r["observed_reward"] is not None
            and r["observed_reward"] >= 0.5
        ),
        "answer_correct_false_negative": lambda r: (
            r["independent_correctness"]
            and r["observed_reward"] is not None
            and r["observed_reward"] < 0.5
        ),
        "high_disagreement": lambda r: r["disagreement"] >= 0.4,
    }
    chosen: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for path in sorted(root.glob("seed-*/level-*/*/evaluation/evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = path.parents[1].name
        for record in payload["records"]:
            for category, predicate in categories.items():
                key = f"{method}/{category}"
                if predicate(record):
                    counts[key] = counts.get(key, 0) + 1
                    if key not in chosen:
                        chosen[key] = {"source": path.relative_to(root).as_posix(), **record}
    write_json(
        root / "failure_analysis.json",
        {
            "selection": "first eligible per method/category in sorted artifact order",
            "counts": counts,
            "examples": chosen,
            "interpretation": (
                "Answer correctness is not proof of reasoning validity; "
                "synthetic false positives are not proof of learned exploitation."
            ),
        },
    )
    lines = [
        "# Qualitative examples",
        "",
        "Deterministic first eligible example per method/category. "
        "All methods are included; unavailable categories are explicitly listed. "
        "These labels concern final answers, not reasoning validity. "
        "Base duplicates are rescoring of the same responses.",
        "",
    ]
    for method in sorted(frame["method"].unique()):
        for category in categories:
            key = f"{method}/{category}"
            lines += [f"## {key}", ""]
            if key not in chosen:
                lines += ["No qualifying measured example.", ""]
                continue
            record = chosen[key]
            response = record["response"]
            excerpt = (
                response if len(response) <= 800 else response[:400] + "\n[…]\n" + response[-400:]
            )
            lines += [
                f"Source: `{record['source']}`; ID: `{record['example_id']}`.",
                "",
                record["metadata"].get("question", "Question unavailable in legacy record."),
                "",
                f"Reference: {record['metadata'].get('reference')}; "
                f"extracted: {record['final_answer']}; "
                f"observed reward: {record['observed_reward']}; "
                f"disagreement: {record['disagreement']}.",
                "",
                "```text",
                excerpt,
                "```",
                "",
            ]
    (root / "failure_analysis.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "runs": len(frame),
        "failures": len(summary["failures"]),
        "aggregate_rows": len(aggregate),
    }
