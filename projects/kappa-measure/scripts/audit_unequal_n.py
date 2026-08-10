"""Scoped audit: comparisons at unequal n and low kappa/p.

This is the one surviving attack surface from the Project 6 brief.  At
kappa/p = 0.05 with n_A = 70 vs n_B = 700 and *identical* true kappa, the
common upward finite-sample bias differs enough across arms to produce a
spurious kappa-hat ratio of ~3.01.  The audit demonstrates the artifact by
Monte Carlo and then shows the vmf-measure verdict: the comparison is
"not resolvable as reported at the original n" — never "wrong".

Also runs the matched-n baselines that must survive:
- identical true kappa, equal n: ratio ~1 (no artifact);
- true ratio 2.0 at equal n in a resolvable cell: reads slightly *below* 2
  (the bias shrinks gaps, it does not inflate them).

Writes ``projects/kappa-measure/results/audit_verdicts.csv`` (schema frozen
in SCHEMA.md, pre-registration commit 2f146ef).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parents[1]))

import vmf_measure  # noqa: E402
from vmf_measure import (  # noqa: E402
    REFUSAL_STATUS,
    kappa_hat,
    mle_kappa,
    rel_rmse_surface,
)
from src.vmf_numpy_hh import NumpyvMFHH  # noqa: E402

VERDICT_NOT_RESOLVABLE = "not resolvable as reported at the original n"
VERDICT_RESOLVABLE = "resolvable"
VERDICT_OUT_OF_SCOPE = "out of scope"

RESULTS = PROJECT_ROOT / "results"


def simulate_arm(p: int, kappa: float, n: int, reps: int, seed: int) -> np.ndarray:
    """kappa-hat over `reps` independent datasets of n samples."""

    sampler = NumpyvMFHH(dim=p, kappa=kappa, seed=seed)
    r_bars = np.empty(reps)
    chunk = max(1, int(2e8 / (n * p)))
    done = 0
    while done < reps:
        take = min(chunk, reps - done)
        draws = sampler.sample((take, n))
        r_bars[done : done + take] = np.linalg.norm(np.mean(draws, axis=1), axis=-1)
        done += take
    return mle_kappa(r_bars, p), r_bars


def audit_comparison(
    claim_id: str,
    comparison_type: str,
    p: int,
    kappa_a: float,
    kappa_b: float,
    n_a: int,
    n_b: int,
    reps: int,
    seed: int,
) -> dict:
    kh_a, rb_a = simulate_arm(p, kappa_a, n_a, reps, seed)
    kh_b, rb_b = simulate_arm(p, kappa_b, n_b, reps, seed + 1)

    # Tool verdict at each arm's observed mean Rbar and original n.
    est_a = kappa_hat(float(np.mean(rb_a)), p, n_a)
    est_b = kappa_hat(float(np.mean(rb_b)), p, n_b)
    refused = est_a.status == REFUSAL_STATUS or est_b.status == REFUSAL_STATUS
    verdict = VERDICT_NOT_RESOLVABLE if refused else VERDICT_RESOLVABLE

    return {
        "claim_id": claim_id,
        "instrument": "kappa_hat_joint_mle",
        "comparison_type": comparison_type,
        "p": p,
        "kappa_true_a": kappa_a,
        "kappa_true_b": kappa_b,
        "kappa_over_p": min(kappa_a, kappa_b) / p,
        "n_a": n_a,
        "n_b": n_b,
        "kappa_hat_a": float(np.mean(kh_a)),
        "kappa_hat_b": float(np.mean(kh_b)),
        "ratio_observed": float(np.mean(kh_a) / np.mean(kh_b)),
        "ratio_true": kappa_a / kappa_b,
        "verdict": verdict,
        "null": "uniform",
        "reps": reps,
        "seed": seed,
        "tool_version": vmf_measure.__version__,
        # extras for the report (not part of the frozen schema; appended)
        "rel_rmse_a": float(rel_rmse_surface(float(np.mean(rb_a)), p, n_a)),
        "rel_rmse_b": float(rel_rmse_surface(float(np.mean(rb_b)), p, n_b)),
        "r_bar_a": float(np.mean(rb_a)),
        "r_bar_b": float(np.mean(rb_b)),
        "kappa_hat_a_sd": float(np.std(kh_a)),
        "kappa_hat_b_sd": float(np.std(kh_b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=90501)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=RESULTS / "audit_verdicts.csv")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    p = 512
    rows = []

    # Canonical attack cell: identical true kappa, unequal n, kappa/p = 0.05.
    rows.append(
        audit_comparison(
            "canonical-unequal-n",
            "unequal_n",
            p, kappa_a=0.05 * p, kappa_b=0.05 * p,
            n_a=70, n_b=700,
            reps=args.reps, seed=args.seed,
        )
    )
    print(rows[-1]["claim_id"], "ratio:", f"{rows[-1]['ratio_observed']:.3f}", flush=True)

    # Control: identical true kappa, matched n (must show no artifact).
    rows.append(
        audit_comparison(
            "control-matched-n",
            "matched_n_baseline",
            p, kappa_a=0.05 * p, kappa_b=0.05 * p,
            n_a=700, n_b=700,
            reps=args.reps, seed=args.seed + 100,
        )
    )
    print(rows[-1]["claim_id"], "ratio:", f"{rows[-1]['ratio_observed']:.3f}", flush=True)

    # Gap-shrinking baseline: true ratio 2.0, matched n, resolvable cell.
    rows.append(
        audit_comparison(
            "control-true-ratio-2",
            "matched_n_baseline",
            p, kappa_a=2.0 * p, kappa_b=1.0 * p,
            n_a=1000, n_b=1000,
            reps=args.reps, seed=args.seed + 200,
        )
    )
    print(rows[-1]["claim_id"], "ratio:", f"{rows[-1]['ratio_observed']:.3f}", flush=True)

    # Brief's quoted illustration: true ratio 2.0 reads as ~1.267 at n=100.
    rows.append(
        audit_comparison(
            "brief-illustration-n100",
            "matched_n_baseline",
            p, kappa_a=0.10 * p, kappa_b=0.05 * p,
            n_a=100, n_b=100,
            reps=args.reps, seed=args.seed + 300,
        )
    )
    print(rows[-1]["claim_id"], "ratio:", f"{rows[-1]['ratio_observed']:.3f}", flush=True)

    # Correction to the brief: at kappa/p = 0.05 the measured spurious ratio
    # is 2.09, not 3.01.  The artifact magnitude interpolates between
    # sqrt(n_B/n_A) at the uniform null and 1 at large kappa/p; 3.01 is
    # reached only near kappa/p ~ 0.017.  Sweep to document this.
    for i, kop in enumerate((0.001, 0.005, 0.013, 0.02, 0.05, 0.1, 0.2)):
        rows.append(
            audit_comparison(
                f"sweep-kop-{kop:g}",
                "unequal_n",
                p, kappa_a=kop * p, kappa_b=kop * p,
                n_a=70, n_b=700,
                reps=args.reps, seed=args.seed + 400 + i,
            )
        )
        print(rows[-1]["claim_id"], "ratio:", f"{rows[-1]['ratio_observed']:.3f}", flush=True)

    fieldnames = [
        "claim_id", "instrument", "comparison_type", "p", "kappa_true_a",
        "kappa_true_b", "kappa_over_p", "n_a", "n_b", "kappa_hat_a",
        "kappa_hat_b", "ratio_observed", "ratio_true", "verdict", "null",
        "reps", "seed", "tool_version",
        "rel_rmse_a", "rel_rmse_b", "r_bar_a", "r_bar_b",
        "kappa_hat_a_sd", "kappa_hat_b_sd",
    ]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {args.out}")
    for row in rows:
        print(
            f"  {row['claim_id']}: ratio {row['ratio_observed']:.3f} "
            f"(true {row['ratio_true']:.2f}) verdict={row['verdict']}"
        )


if __name__ == "__main__":
    main()
