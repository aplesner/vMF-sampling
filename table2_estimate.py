from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special


def _ap_ratio(kappa: float, p: int) -> float:
    order = 0.5 * p
    num = special.iv(order, kappa)
    den = special.iv(order - 1.0, kappa)
    return float(num / den)


def _banerjee(Rbar: float, p: int) -> float:
    return Rbar * (p - Rbar**2) / (1.0 - Rbar**2)


def _tanabe(Rbar: float, p: int) -> float:
    k_l = Rbar * (p - 2.0) / (1.0 - Rbar**2)
    k_u = Rbar * p / (1.0 - Rbar**2)

    def phi(kappa: float) -> float:
        return Rbar * kappa / _ap_ratio(kappa, p)

    phi_ku = phi(k_u)
    phi_kl = phi(k_l)
    denom = phi_ku - phi_kl
    if denom == 0:
        return float("nan")
    return (k_l * phi_ku - k_u * phi_kl) / denom - (k_u - k_l)


def _newton(Rbar: float, p: int) -> float:
    kappa = _banerjee(Rbar, p)
    for _ in range(2):
        ap = _ap_ratio(kappa, p)
        denom = 1.0 - ap**2 - (p - 1.0) / kappa * ap
        kappa = kappa - (ap - Rbar) / denom
    return kappa


def _compute_row(row: pd.Series) -> dict[str, float]:
    p = int(row["p"])
    kappa_true = float(row["kappa_true"])
    rbar = float(row["rbar"])
    return {
        "p": float(p),
        "kappa_true": float(kappa_true),
        "seed": float(row.get("seed", np.nan)),
        "num_samples": float(row.get("num_samples", np.nan)),
        "rbar": float(rbar),
        "banerjee_error": float(np.abs(_banerjee(rbar, p) - kappa_true)),
        "tanabe_error": float(np.abs(_tanabe(rbar, p) - kappa_true)),
        "newton_error": float(np.abs(_newton(rbar, p) - kappa_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Table 2 errors from sampled Rbar.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with sampled Rbar.")
    parser.add_argument("--output", type=Path, required=True, help="CSV output for errors.")
    parser.add_argument("--summary-output", type=Path, help="Optional aggregated summary CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rows = [_compute_row(row) for _, row in df.iterrows()]
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    if args.summary_output is not None:
        summary = result.groupby(["p", "kappa_true"]).agg(
            banerjee_error_mean=("banerjee_error", "mean"),
            banerjee_error_std=("banerjee_error", "std"),
            tanabe_error_mean=("tanabe_error", "mean"),
            tanabe_error_std=("tanabe_error", "std"),
            newton_error_mean=("newton_error", "mean"),
            newton_error_std=("newton_error", "std"),
            n=("seed", "count"),
        )
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.reset_index().to_csv(args.summary_output, index=False)


if __name__ == "__main__":
    main()
