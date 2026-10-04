"""Create automatically measured result tables and qualitative cases."""

import argparse
import json

from src.experiments.reporting import create_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    print(json.dumps(create_report(args.results), indent=2))
