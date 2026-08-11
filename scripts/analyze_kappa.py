"""Summarize the kappa-estimator benchmark produced by ``benchmark_kappa.py``.

Emits Markdown tables for the three studies plus the sampler-cost table, and
writes tidy summary CSVs next to the raw measurement files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load(directory: Path, name: str, tag: str) -> pd.DataFrame | None:
    suffix = f"_{tag}" if tag else ""
    path = directory / f"{name}{suffix}.csv"
    return pd.read_csv(path) if path.exists() else None


def _scientific(value: object) -> str:
    if not isinstance(value, float):
        return str(value)
    if pd.isna(value):
        return "fail"
    return f"{value:.1e}"


def _markdown(frame: pd.DataFrame) -> str:
    return frame.map(_scientific).to_markdown()


def population_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Relative error against the generating kappa, by method."""

    return frame.pivot_table(
        index=["dimension", "kappa_true"],
        columns="method",
        values="rel_error",
        dropna=False,
    )


def failure_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Count non-finite or grossly wrong estimates per method and dimension."""

    working = frame.assign(
        broken=lambda f: f["kappa_hat"].isna() | (f["rel_error"] > 1e-3)
    )
    return working.pivot_table(
        index="method", columns="dimension", values="broken", aggfunc="sum"
    ).astype(int)


def sampled_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numerical error against the sample MLE, and statistical error vs truth."""

    numerical = frame.pivot_table(
        index=["dimension", "kappa_true"],
        columns="method",
        values="numerical_rel_error",
        aggfunc="median",
        dropna=False,
    )
    statistical = frame.pivot_table(
        index=["dimension", "kappa_true"],
        columns="method",
        values="statistical_rel_error",
        aggfunc="median",
        dropna=False,
    )
    return numerical, statistical


def timing_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot_table(
        index="method", columns="batch_size", values="estimates_per_second", dropna=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("measurements/kappa"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    lines: list[str] = ["# vMF concentration-estimator benchmark", ""]

    environment_path = args.input_dir / f"environment{suffix}.json"
    if environment_path.exists():
        record = json.loads(environment_path.read_text())
        lines += ["## Environment", "", "```json", json.dumps(record, indent=2), "```", ""]

    population = _load(args.input_dir, "population", args.tag)
    if population is not None:
        table = population_table(population)
        lines += [
            "## 1. Numerical approximation (population `r_bar`)",
            "",
            "Relative error against the generating kappa. Exact solvers should"
            " reach machine precision; `fail` marks a non-finite estimate.",
            "",
            _markdown(table),
            "",
            "### Failures and non-convergence (relative error > 1e-3 or NaN)",
            "",
            failure_table(population).to_markdown(),
            "",
        ]
        table.to_csv(args.input_dir / f"summary_population{suffix}.csv")

    sampled = _load(args.input_dir, "sampled", args.tag)
    if sampled is not None:
        numerical, statistical = sampled_table(sampled)
        lines += [
            "## 2. Estimation from sampled data",
            "",
            "Median relative error against the high-precision MLE for the *same*"
            " sample. This is solver error only.",
            "",
            _markdown(numerical),
            "",
            "Median relative error against the generating kappa. This includes"
            " genuine finite-sample bias shared by all accurate solvers.",
            "",
            _markdown(statistical),
            "",
        ]
        numerical.to_csv(args.input_dir / f"summary_sampled_numerical{suffix}.csv")
        statistical.to_csv(args.input_dir / f"summary_sampled_statistical{suffix}.csv")

    timing = _load(args.input_dir, "timing", args.tag)
    if timing is not None:
        table = timing_table(timing)
        lines += [
            "## 3. Throughput (estimates per second)",
            "",
            f"Dimension {int(timing['dimension'].iloc[0])}, device"
            f" `{timing['device'].iloc[0]}`. Sample generation and the reduction"
            " to `r_bar` are excluded.",
            "",
            table.map(lambda v: "n/a" if pd.isna(v) else f"{v:,.0f}").to_markdown(),
            "",
        ]
        table.to_csv(args.input_dir / f"summary_timing{suffix}.csv")

    reliability = _load(args.input_dir, "reliability", args.tag)
    if reliability is not None:
        table = reliability.pivot_table(
            index=["dimension", "kappa_true"],
            columns="num_samples",
            values="median_rel_error",
        )
        # Smallest tested sample count reaching each accuracy target.
        requirements = []
        for (dimension, kappa), group in reliability.groupby(
            ["dimension", "kappa_true"]
        ):
            ordered = group.sort_values("num_samples")
            row: dict[str, object] = {"dimension": dimension, "kappa_true": kappa}
            for tolerance, label in ((0.10, "n for 10%"), (0.01, "n for 1%")):
                reached = ordered.loc[ordered["median_rel_error"] <= tolerance]
                row[label] = (
                    f"{int(reached['num_samples'].iloc[0]):,}"
                    if not reached.empty
                    else f"> {int(ordered['num_samples'].max()):,}"
                )
            requirements.append(row)
        lines += [
            "## 5. How much data the concentration MLE needs",
            "",
            "Median relative error against the generating kappa, by sample count."
            " Every accurate solver gives the same value here, so this is a"
            " property of the estimation problem, not of the numerical method.",
            "",
            _markdown(table),
            "",
            "Smallest tested sample count reaching each accuracy target:",
            "",
            pd.DataFrame(requirements).to_markdown(index=False),
            "",
        ]
        table.to_csv(args.input_dir / f"summary_reliability{suffix}.csv")

    cost = _load(args.input_dir, "sampler_cost", args.tag)
    if cost is not None:
        table = cost.pivot_table(
            index="dimension", columns="backend", values="speedup_over_scipy"
        )
        lines += [
            "## 4. Cost of generating the study's own data",
            "",
            "Sampler throughput speedup over the SciPy reference, measured on the"
            " same random mean directions used by study 2.",
            "",
            table.map(lambda v: f"{v:.1f}x").to_markdown(),
            "",
        ]

    report = "\n".join(lines)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
