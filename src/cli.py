"""Command-line entry point for CPU audits, GRPO training, and plotting."""

import argparse
import json
import logging

from src.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "train", "experiment", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "train":
            command.add_argument("--resume")
    plot = commands.add_parser("plot")
    plot.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    try:
        if args.command == "plot":
            from src.visualization.plots import plot_results

            print(json.dumps([str(path) for path in plot_results(args.results)], indent=2))
            return 0
        config = load_config(args.config)
        if args.command == "validate":
            print(json.dumps(config, indent=2))
        elif args.command == "evaluate":
            from src.evaluation.runner import evaluate

            print(json.dumps(evaluate(config)["metrics"], indent=2))
        elif args.command == "train":
            from src.training.runner import train

            print(json.dumps(train(config, args.resume)["metrics"], indent=2))
        else:
            from src.experiments.runner import experiment

            result = experiment(config)
            print(json.dumps({"completed": len(result["runs"]), "failed": len(result["failures"])}, indent=2))
            return 1 if result["failures"] else 0
        return 0
    except (ValueError, OSError, ImportError) as error:
        logging.error("%s: %s", type(error).__name__, error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
