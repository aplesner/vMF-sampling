"""T1a — the reference (SciPy-based) S-VAE is numerically invalid at m >= 1024.

Reproduces the exact numerics of Davidson et al.'s reference implementation:
  KL      = kappa * ive(m/2, k)/ive(m/2-1, k) + (m/2-1) log k - (m/2) log 2pi
            - log ive(m/2-1, k) - kappa + log S_{m-1}
  dKL/dk  = their eq. 6, built from ive ratios ive(m/2+1)/ive(m/2-1) etc.

At order >~ 511 scipy's ive underflows to exactly 0, so the KL forward pass
goes to +inf and the eq.-6 gradient is 0/0 = NaN — the loss is non-finite
before training even starts, at kappa values an S-VAE encoder actually emits
(softplus init ~ 0.54, trend values kappa~ 3..20).  Ours (CUSF/torch
log-Bessel) stays finite with 1e-12-level agreement where SciPy still works.

Writes results/t1_numerical_validity.csv.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import special

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vmf_kl import kl_grad_kappa, kl_vmf_uniform  # noqa: E402


def reference_kl(kappa: float, m: int) -> float:
    """Davidson reference KL, scipy ive (as in the public s-vae code)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = special.ive(m / 2, kappa) / special.ive(m / 2 - 1, kappa)
        log_c = (
            (m / 2 - 1) * np.log(kappa)
            - (m / 2) * np.log(2 * np.pi)
            - np.log(special.ive(m / 2 - 1, kappa))
            - kappa
        )
        log_s = math.log(2) + (m / 2) * math.log(math.pi) - math.lgamma(m / 2)
        return kappa * ratio + log_c + log_s


def reference_kl_grad(kappa: float, m: int) -> float:
    """Davidson eq. 6 with ive ratios (their gradient path).

    eq. 6 simplifies to kappa * A_m'(kappa) =
    (k/2) (1 + I_{m/2+1}/I_{m/2-1} - A (I_{m/2-2} + I_{m/2})/I_{m/2-1});
    the ive ratio version is what their code computes, and it is 0/0 at
    order >~ 511.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a = special.ive(m / 2, kappa) / special.ive(m / 2 - 1, kappa)
        r1 = special.ive(m / 2 + 1, kappa) / special.ive(m / 2 - 1, kappa)
        r2 = (special.ive(m / 2 - 2, kappa) + special.ive(m / 2, kappa)) / special.ive(
            m / 2 - 1, kappa
        )
        return 0.5 * kappa * (1 + r1 - a * r2)


def main() -> None:
    rows = []
    print(f"{'m':>6} {'kappa':>10} {'ref KL':>14} {'ref dKL':>12} {'our KL':>14} {'our dKL':>12}")
    for m in [3, 41, 128, 512, 1024, 4096]:
        for kappa in [0.54, 3.0, 20.0, 100.0, 500.0]:
            ref = reference_kl(kappa, m)
            ref_g = reference_kl_grad(kappa, m)
            ours = kl_vmf_uniform(kappa, m).item()
            ours_g = kl_grad_kappa(kappa, m).item()
            rows.append(
                dict(
                    m=m,
                    kappa=kappa,
                    reference_kl=ref,
                    reference_grad=ref_g,
                    ours_kl=ours,
                    ours_grad=ours_g,
                    reference_finite=bool(np.isfinite(ref) and np.isfinite(ref_g)),
                    ours_finite=bool(np.isfinite(ours) and np.isfinite(ours_g)),
                )
            )
            if kappa in (0.54, 20.0, 500.0):
                print(f"{m:>6} {kappa:>10} {ref:>14.6g} {ref_g:>12.4g} {ours:>14.6g} {ours_g:>12.4g}")

    out = PROJECT_ROOT / "results" / "t1_numerical_validity.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Agreement where scipy still works (m <= 512, orders < 511).
    ok = [r for r in rows if r["m"] <= 512 and r["reference_finite"]]
    max_rel = max(
        abs(r["reference_kl"] - r["ours_kl"]) / max(abs(r["ours_kl"]), 1e-30) for r in ok
    )
    broken = [r for r in rows if r["m"] >= 1024 and not r["reference_finite"]]
    ours_ok = all(r["ours_finite"] for r in rows)
    print(f"\nmax |rel err| vs scipy where scipy valid: {max_rel:.3e}")
    print(f"non-finite reference rows at m>=1024: {len(broken)}/{sum(1 for r in rows if r['m']>=1024)}")
    print(f"all ours finite: {ours_ok}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
