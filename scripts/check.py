"""Run the same CPU-safe quality gates as CI, stopping on first failure."""

import subprocess
import sys


def main() -> None:
    for args in (
        ["ruff", "check", "src", "tests", "scripts"],
        ["ruff", "format", "--check", "src", "tests", "scripts"],
        ["mypy", "src"],
        ["pytest", "-q"],
    ):
        subprocess.run([sys.executable, "-m", *args], check=True)


if __name__ == "__main__":
    main()
