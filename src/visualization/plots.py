"""Seed means and empirical standard deviations, with individual run visibility."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(results: str | Path) -> list[Path]:
    root = Path(results)
    paths = [root] if root.is_file() else sorted(root.rglob("summary.csv"))
    if not paths:
        raise ValueError("No summary.csv found; run an experiment first")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"method", "corruption_level", "independent_accuracy", "reward_hacking_gap"}
    if not required <= set(frame.columns):
        raise ValueError(f"Results lack columns: {required - set(frame.columns)}")
    if "corruption_kind" in frame and frame["corruption_kind"].nunique() > 1:
        raise ValueError("Plot each corruption kind separately to avoid confounded averages")
    if "execution_kind" in frame and frame["execution_kind"].nunique() > 1:
        raise ValueError("Do not pool fixture and research results")
    out = (root.parent if root.is_file() else root) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    fixture = "execution_kind" in frame and frame["execution_kind"].str.contains("fixture").any()
    suffix = " (software fixture; not research evidence)" if fixture else ""
    if not fixture:
        metadata_path = paths[0].parent / "manifest.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            label = metadata.get("config", {}).get("experiment", {}).get("label", "")
            if "smoke" in label:
                suffix = "\nPretrained pipeline smoke; not capability evidence"
            elif "preliminary" in label:
                suffix = "\nPreliminary experiment; limited training and test budget"
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    panels = [
        ("independent_accuracy", "Independent correctness", "robustness"),
        ("reward_hacking_gap", "Reward − paired correctness", "reward_hacking"),
        ("observed_reward", "Observed reward", "observed_reward"),
        ("verifier_disagreement", "Verifier disagreement", "disagreement_corruption"),
        ("extraction_failure_rate", "Extraction failure rate", "extraction_failure"),
    ]
    for metric, label, name in panels:
        if metric not in frame:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for method, subset in frame.groupby("method"):
            grouped = subset.groupby("corruption_level")[metric].agg(["mean", "std"])
            x = grouped.index.to_numpy(dtype=float) * 100
            y = grouped["mean"].to_numpy(dtype=float)
            std = grouped["std"].fillna(0).to_numpy(dtype=float)
            ax.plot(x, y, marker="o", label=str(method))
            ax.fill_between(x, y - std, y + std, alpha=0.15)
        ax.set(xlabel="Reward corruption (%)", ylabel=label, title=label + suffix)
        ax.set_xticks(sorted(frame["corruption_level"].unique() * 100))
        if metric != "reward_hacking_gap":
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.2)
        for ext in ("png", "pdf"):
            path = out / f"{name}.{ext}"
            fig.savefig(path)
            generated.append(path)
        plt.close(fig)
    for x_metric, y_metric, name in [
        ("observed_reward", "independent_accuracy", "reward_correctness"),
        ("verifier_disagreement", "independent_accuracy", "disagreement"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for method, subset in frame.groupby("method"):
            ax.scatter(subset[x_metric], subset[y_metric], label=str(method), alpha=0.7)
        ax.set(
            xlabel=x_metric.replace("_", " "), ylabel="Independent correctness", title=name + suffix
        )
        ax.set_ylim(0, 1)
        ax.legend()
        path = out / f"{name}.png"
        fig.savefig(path)
        generated.append(path)
        plt.close(fig)
    ks = sorted(
        [str(column) for column in frame if str(column).startswith("pass@")],
        key=lambda c: int(c[5:]),
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for (sampling_method, level), subset in frame.groupby(["method", "corruption_level"]):
        ax.errorbar(
            [int(k[5:]) for k in ks],
            [subset[k].mean() for k in ks],
            yerr=[subset[k].std(ddof=1) if len(subset) > 1 else 0 for k in ks],
            capsize=3,
            marker="o",
            label=f"{sampling_method}, corruption={level}",
        )
    ax.set(xlabel="Sampling budget k", ylabel="pass@k", title="Sampling attribution" + suffix)
    ax.set_ylim(0, 1)
    ax.legend()
    path = out / "sampling.png"
    fig.savefig(path)
    generated.append(path)
    plt.close(fig)
    return generated
