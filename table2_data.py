from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import special


PAIRS = [
    (500, 100),
    (500, 500),
    (500, 1000),
    (500, 5000),
    (500, 10000),
    (500, 20000),
    (500, 50000),
    (500, 100000),
    (1000, 100),
    (1000, 500),
    (1000, 1000),
    (1000, 5000),
    (1000, 10000),
    (1000, 20000),
    (1000, 50000),
    (1000, 100000),
    (5000, 100),
    (5000, 500),
    (5000, 1000),
    (5000, 5000),
    (5000, 10000),
    (5000, 20000),
    (5000, 50000),
    (5000, 100000),
    (10000, 100),
    (10000, 500),
    (10000, 1000),
    (10000, 5000),
    (10000, 10000),
    (10000, 20000),
    (10000, 50000),
    (10000, 100000),
    (20000, 100),
    (20000, 500),
    (20000, 1000),
    (20000, 5000),
    (20000, 10000),
    (20000, 20000),
    (20000, 50000),
    (20000, 100000),
    (100000, 100),
    (100000, 500),
    (100000, 1000),
    (100000, 5000),
    (100000, 10000),
    (100000, 20000),
    (100000, 50000),
    (100000, 100000),
]


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


def _write_csv_row(path: Path, row: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate Table 2 errors from Sra (2011).")
    parser.add_argument("--output", type=Path, help="CSV output path for incremental results.")
    parser.add_argument("--summary", action="store_true", help="Print results at end.")
    args = parser.parse_args()

    rows: list[dict[str, float]] | None = [] if (args.summary or args.output is None) else None

    for p, kappa_true in PAIRS:
        Rbar = _ap_ratio(float(kappa_true), int(p))
        banerjee = _banerjee(Rbar, int(p))
        tanabe = _tanabe(Rbar, int(p))
        newton = _newton(Rbar, int(p))
        row = {
            "p": float(p),
            "kappa_true": float(kappa_true),
            "banerjee_error": float(np.abs(banerjee - kappa_true)),
            "tanabe_error": float(np.abs(tanabe - kappa_true)),
            "newton_error": float(np.abs(newton - kappa_true)),
        }
        if args.output is not None:
            _write_csv_row(args.output, row)
        if rows is not None:
            rows.append(row)

    if rows is None:
        return

    for row in rows:
        print(
            f"({int(row['p'])}, {int(row['kappa_true'])}) "
            f"{row['banerjee_error']:.2e} "
            f"{row['tanabe_error']:.2e} "
            f"{row['newton_error']:.2e}"
        )


if __name__ == "__main__":
    main()
