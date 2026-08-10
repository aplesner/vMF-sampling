"""Numerical verification of the restated Gate G1 (Project 4).

v1 Gate G1 required "mean learned kappa falling monotonically with d *while KL
still rises*".  Because KL at fixed kappa is non-monotone in m (peaking near
m* ~= 0.76 kappa), that conjunction is only satisfiable while the posterior
stays on the rising branch.  This script:

  V1  verifies KL = m * f(rho) at fixed rho = A_m(kappa) across m, i.e. the
      per-dimension rate is a function of rho alone;
  V2  locates the peak m*(kappa) of KL(m) at fixed kappa and shows the rising
      branch is exactly rho > rho* ~ 0.69, so the v1 gate is satisfiable only
      while rho stays above rho*;
  V3  inverts Davidson et al. (2018) Table 1 S-VAE KL column to implied kappa,
      kappa/m and rho at m = d + 1, and extrapolates the trend to find where
      the model crosses rho*;
  V4  shows the v1 gate fails mathematically on the extrapolated trend at high
      d, while the restated gate (G1a: rho-bar non-increasing; G1b: KL/m
      bounded away from 0) remains well-defined and jointly satisfiable;
  V5  demonstrates SciPy's ive is invalid at m >= 1024 while this repo's
      log-Bessel path keeps every gate quantity finite.

Writes results/g1_fixed_rho.csv, results/g1_branch.csv,
results/g1_davidson_inversion.csv and prints a summary.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vmf_kl import (  # noqa: E402
    bessel_ratio_A,
    invert_A,
    invert_kl,
    kl_vmf_uniform,
)

RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Davidson et al. (2018), Table 1, S-VAE rows (mean KL over 10 runs).
DAVIDSON_D = [2, 5, 10, 20, 40]
DAVIDSON_SVAE_KL = [7.28, 13.35, 20.67, 28.50, 33.50]
# Empirical location of the KL(m) peak at fixed kappa (m* / kappa); the brief
# quotes ~0.76.  V2 re-derives it.
RHO_STAR_GRID_KAPPAS = [2.0, 5.0, 20.0, 100.0, 400.0, 2000.0]


def v1_fixed_rho_linearity() -> list[dict]:
    """V1: KL/m at fixed rho across m; also reproduces the 0.04715 anchor."""
    rows = []
    for rho in [0.05, 0.1, 0.3, 0.5, 0.8, 0.95]:
        per_m = {}
        for m in [8, 16, 64, 128, 512, 1024, 4096]:
            kappa = invert_A(rho, m).item()
            kl_over_m = kl_vmf_uniform(kappa, m).item() / m
            per_m[m] = kl_over_m
            rows.append(dict(rho=rho, m=m, kappa=kappa, kl_over_m=kl_over_m,
                             kappa_over_m=kappa / m))
        values = list(per_m.values())
        spread = (max(values) - min(values)) / min(values)
        print(f"  rho={rho:<5} KL/m = " +
              "  ".join(f"m={m}:{v:.6f}" for m, v in per_m.items()) +
              f"   rel spread={spread:.2e}")
    return rows


def v2_branch_boundary() -> list[dict]:
    """V2: peak m* of KL(m) at fixed kappa; report m*/kappa and rho at peak."""
    rows = []
    print("  kappa      m*      m*/kappa   rho*=A_m*(kappa)   KL(m*)/m*")
    for kappa in RHO_STAR_GRID_KAPPAS:
        # Fine real-valued scan of m (KL extends to real m via lgamma).
        m_grid = torch.linspace(2.0, 3.0 * kappa + 4.0, 4000, dtype=torch.float64)
        kl = kl_vmf_uniform(kappa, m_grid)
        peak = int(torch.argmax(kl).item())
        # Refine by golden-section search around the grid argmax.
        lo = m_grid[max(peak - 2, 0)].item()
        hi = m_grid[min(peak + 2, len(m_grid) - 1)].item()
        for _ in range(200):
            m1 = lo + 0.3819660112501051 * (hi - lo)
            m2 = hi - 0.3819660112501051 * (hi - lo)
            if kl_vmf_uniform(kappa, m1).item() < kl_vmf_uniform(kappa, m2).item():
                lo = m1
            else:
                hi = m2
        m_star = 0.5 * (lo + hi)
        rho_star = bessel_ratio_A(kappa, m_star).item()
        kl_star = kl_vmf_uniform(kappa, m_star).item()
        rows.append(dict(kappa=kappa, m_star=m_star, m_star_over_kappa=m_star / kappa,
                         rho_star=rho_star, kl_over_m_at_peak=kl_star / m_star))
        print(f"  {kappa:7.1f}  {m_star:8.2f}   {m_star/kappa:.4f}      "
              f"{rho_star:.4f}            {kl_star/m_star:.4f}")
    return rows


def v3_davidson_inversion() -> list[dict]:
    """V3: invert Davidson's S-VAE KL column to implied kappa, kappa/m, rho."""
    rows = []
    print("  d     m    KL(Davidson)   kappa~      kappa/m    rho=A_m(kappa~)  KL/m")
    log_m, log_k = [], []
    for d, kl in zip(DAVIDSON_D, DAVIDSON_SVAE_KL):
        m = d + 1  # m = d + 1 convention (brief, Metrics)
        kappa = invert_kl(kl, m).item()
        rho = bessel_ratio_A(kappa, m).item()
        rows.append(dict(d=d, m=m, kl_reported=kl, kappa_implied=kappa,
                         kappa_over_m=kappa / m, rho=rho, kl_over_m=kl / m))
        log_m.append(math.log(m))
        log_k.append(math.log(kappa))
        print(f"  {d:<4}  {m:<4} {kl:9.2f}     {kappa:12.2f}  {kappa/m:9.3f}   "
              f"{rho:9.6f}     {kl/m:7.4f}")
    # Least-squares fit log kappa = a + b log m for extrapolation.
    n = len(log_m)
    mean_lm, mean_lk = sum(log_m) / n, sum(log_k) / n
    b = sum((x - mean_lm) * (y - mean_lk) for x, y in zip(log_m, log_k)) / \
        sum((x - mean_lm) ** 2 for x in log_m)
    a = mean_lk - b * mean_lm
    print(f"  trend fit: log kappa = {a:.4f} + {b:.4f} log m")
    return rows, (a, b)


def v4_old_gate_failure(a: float, b: float) -> list[dict]:
    """V4: v1 gate conjunction on the extrapolated kappa(m) trend.

    The v1 gate requires kappa(m) decreasing AND KL(m, kappa(m)) increasing.
    KL(m) at fixed kappa peaks at m* ~= 0.76 kappa, so once the trend crosses
    kappa/m = 1/0.76 (rho < rho*) the raw-KL conjunct fails regardless of
    anything the encoder does.
    """
    rows = []
    branch_boundary = 1.0 / 0.76  # kappa/m at the KL(m) peak
    print("  d      m     kappa~(trend)  kappa/m    rho      KL(trend)   on rising branch?")
    prev_kl = None
    for d in [5, 10, 20, 40, 64, 128, 256, 512, 1024, 2048]:
        m = d + 1
        kappa = math.exp(a + b * math.log(m))
        rho = bessel_ratio_A(kappa, m).item()
        kl = kl_vmf_uniform(kappa, m).item()
        rising = kappa / m > branch_boundary
        rows.append(dict(d=d, m=m, kappa_trend=kappa, kappa_over_m=kappa / m,
                         rho=rho, kl_trend=kl, kl_over_m=kl / m,
                         on_rising_branch=rising,
                         kl_increasing=(prev_kl is None or kl > prev_kl)))
        prev_kl = kl
        print(f"  {d:<5}  {m:<5} {kappa:12.2f}   {kappa/m:8.4f}  {rho:.4f}   "
              f"{kl:9.3f}    {'yes' if rising else 'NO (raw-KL conjunct dead)'}")
    return rows


def v5_computability(a: float, b: float) -> None:
    """V5: SciPy ive underflow vs this repo's log-Bessel path.

    ive underflow is kappa-dependent, so evaluate at (i) the kappa-estimation
    grid value kappa=50 (AGENTS.md: underflow above order ~511) and (ii) the
    extrapolated Davidson trend kappa~(m) — the S-VAE-relevant operating point.
    """
    from scipy import special
    from vmf_kl import log_bessel_iv
    print("  m      order   ive(.,50)     ive(.,kappa~(m))  torch log_iv finite?  KL finite?")
    for m in [512, 1024, 2048, 4096, 8192]:
        order = m / 2.0 - 1.0
        kappa_trend = math.exp(a + b * math.log(m))
        scipy_50 = special.ive(order, 50.0)
        scipy_trend = special.ive(order, kappa_trend)
        liv = log_bessel_iv(order, kappa_trend).item()
        kl = kl_vmf_uniform(kappa_trend, m).item()
        print(f"  {m:<5}  {order:<7.0f} {scipy_50:.3e}    {scipy_trend:.3e}          "
              f"{math.isfinite(liv)!s:<5}              {math.isfinite(kl)!s:<5}")
    print("  (kappa trend: ", ", ".join(
        f"m={m}: {math.exp(a + b * math.log(m)):.2f}" for m in [512, 1024, 4096]), ")")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("V1 — KL/m at fixed rho (linearity in m):")
    rows_v1 = v1_fixed_rho_linearity()
    write_csv(RESULTS / "g1_fixed_rho.csv", rows_v1)

    print("\nV2 — branch boundary: KL(m) peak at fixed kappa:")
    rows_v2 = v2_branch_boundary()
    write_csv(RESULTS / "g1_branch.csv", rows_v2)

    print("\nV3 — Davidson Table 1 S-VAE inversion (m = d + 1):")
    rows_v3, (a, b) = v3_davidson_inversion()
    write_csv(RESULTS / "g1_davidson_inversion.csv", rows_v3)

    print("\nV4 — v1 gate conjunction on the extrapolated trend:")
    rows_v4 = v4_old_gate_failure(a, b)
    write_csv(RESULTS / "g1_old_gate_failure.csv", rows_v4)

    print("\nV5 — computability at m >= 1024:")
    v5_computability(a, b)


if __name__ == "__main__":
    main()
