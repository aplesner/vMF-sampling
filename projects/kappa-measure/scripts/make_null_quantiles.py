"""Generate the frozen-schema null-quantile table (deliverable D2).

Writes ``projects/kappa-measure/results/null_quantiles.csv``:
- closed-form uniform-null quantiles for r_bar, kappa_hat, mean_cosine over
  the (n, p) grid;
- Monte Carlo mean-removed rows at the audit's reference cells, as a
  cross-check on the residual-concentration null.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import vmf_measure  # noqa: E402
from vmf_measure import null_quantile, null_quantile_table  # noqa: E402

P_VALUES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
N_VALUES = [50, 70, 100, 500, 700, 1000, 10000]
QUANTILES = [0.5, 0.9, 0.95, 0.99]

MC_CELLS = [(128, 70), (128, 700), (512, 70), (512, 700)]

RESULTS = PROJECT_ROOT / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = null_quantile_table(P_VALUES, N_VALUES, QUANTILES)
    for p, n in MC_CELLS:
        for statistic in ("r_bar", "kappa_hat"):
            for q in (0.5, 0.95):
                rows.append(
                    null_quantile(
                        statistic, p, n, q, null="mean_removed", reps=2000, seed=616
                    )
                )

    fieldnames = [
        "statistic", "null", "p", "n", "quantile", "value", "method",
        "reps", "seed", "tool_version",
    ]
    out = RESULTS / "null_quantiles.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "statistic": row.statistic,
                    "null": row.null,
                    "p": row.p,
                    "n": row.n,
                    "quantile": row.quantile,
                    "value": row.value,
                    "method": row.method,
                    "reps": row.reps if row.reps is not None else "",
                    "seed": row.seed if row.seed is not None else "",
                    "tool_version": vmf_measure.__version__,
                }
            )
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
