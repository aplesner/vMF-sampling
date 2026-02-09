from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge benchmark CSV files.")
    parser.add_argument(
        "--input",
        default="measurements/benchmark_*.csv",
        help="Glob pattern for input CSVs.",
    )
    parser.add_argument(
        "--output",
        default="measurements/benchmark_all.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    paths = sorted(Path().glob(args.input))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {args.input}")

    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
