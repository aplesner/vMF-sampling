"""Monte Carlo validation of the closed-form rel-RMSE surface (deliverable D1).

For each (p, kappa/p, n/p) grid cell, draw `reps` independent vMF datasets of
n samples with the Householder NumPy sampler, estimate kappa from each, and
compare the empirical relative RMSE against the closed-form surface evaluated
at the population Rbar = A_p(kappa).  Writes
``projects/kappa-measure/results/surface_validation.csv`` (schema frozen in
SCHEMA.md).

Acceptance target (pre-registered in SCHEMA.md): ratio_emp_over_pred within a
factor of 2 across the grid.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parents[1]))

import vmf_measure  # noqa: E402
from vmf_measure import bessel_ratio_ap, mle_kappa, rel_rmse_surface  # noqa: E402
from src.vmf_numpy_hh import NumpyvMFHH  # noqa: E402

P_VALUES = [128, 512, 2048]
KAPPA_OVER_P = [0.05, 0.2, 0.5, 1.0, 3.0]
N_OVER_P = [0.25, 0.5, 2.0, 8.0]

# Element budget per cell: reps adapt so high-dimensional cells stay cheap
# while low-dimensional cells get tight RMSE estimates.
CELL_ELEMENT_BUDGET = 2e9
MAX_REPS = 2000
MIN_REPS = 60
CHUNK_ELEMENTS = 2e8

RESULTS = PROJECT_ROOT / "results"


def run_cell(p: int, kappa: float, n: int, seed: int) -> dict:
    reps = int(np.clip(CELL_ELEMENT_BUDGET / (n * p), MIN_REPS, MAX_REPS))
    chunk = max(1, int(CHUNK_ELEMENTS / (n * p)))
    sampler = NumpyvMFHH(dim=p, kappa=kappa, seed=seed)

    r_bars = np.empty(reps)
    done = 0
    while done < reps:
        take = min(chunk, reps - done)
        draws = sampler.sample((take, n))
        r_bars[done : done + take] = np.linalg.norm(np.mean(draws, axis=1), axis=-1)
        done += take

    kappa_hats = mle_kappa(r_bars, p)
    empirical = float(np.sqrt(np.mean((kappa_hats / kappa - 1.0) ** 2)))
    r_population = float(bessel_ratio_ap(kappa, p))
    predicted = float(rel_rmse_surface(r_population, p, n, kappa=kappa))
    return {
        "p": p,
        "kappa": kappa,
        "kappa_over_p": kappa / p,
        "n": n,
        "n_over_p": n / p,
        "reps": reps,
        "r_bar_mean": float(np.mean(r_bars)),
        "empirical_rel_rmse": empirical,
        "predicted_rel_rmse": predicted,
        "ratio_emp_over_pred": empirical / predicted,
        "seed": seed,
        "sampler": "NumpyvMFHH",
        "tool_version": vmf_measure.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--out", type=Path, default=RESULTS / "surface_validation.csv")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    rows = []
    started = time.time()
    for p in P_VALUES:
        for kop in KAPPA_OVER_P:
            for nop in N_OVER_P:
                kappa = kop * p
                n = max(4, round(nop * p))
                seed = args.seed + hash((p, round(kappa * 10), n)) % 100_000
                t0 = time.time()
                row = run_cell(p, kappa, n, seed)
                rows.append(row)
                print(
                    f"p={p:5d} k/p={kop:4.2f} n/p={nop:5.2f} reps={row['reps']:5d} "
                    f"emp={row['empirical_rel_rmse']:.4g} pred={row['predicted_rel_rmse']:.4g} "
                    f"ratio={row['ratio_emp_over_pred']:.3f} ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ratios = np.array([row["ratio_emp_over_pred"] for row in rows])
    print(f"\nwrote {len(rows)} rows to {args.out} in {time.time() - started:.0f}s")
    print(f"ratio emp/pred: min={ratios.min():.3f} max={ratios.max():.3f} "
          f"(acceptance: all within a factor of 2)")


if __name__ == "__main__":
    main()
