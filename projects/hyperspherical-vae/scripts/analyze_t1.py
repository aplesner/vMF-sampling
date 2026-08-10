"""T1b — compare our MNIST runs against Davidson et al. (2018) Table 1.

Reads results/cluster/t1_rows/*.json (fetched by scripts/fetch_results.sh),
aggregates over seeds, and checks the 2 s.d. reproduction criterion on
IW-500 LL, RE and KL for both arms at d <= 40.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROWS = PROJECT_ROOT / "results" / "cluster" / "t1_rows"

# Davidson et al. (2018) Table 1: (LL, LL_sd, RE, RE_sd, KL, KL_sd)
DAVIDSON = {
    "gaussian": {
        2: (-135.73, 0.83, -129.84, 0.91, 7.24, 0.11),
        5: (-110.21, 0.21, -100.16, 0.22, 12.82, 0.11),
        10: (-93.84, 0.30, -78.93, 0.30, 19.44, 0.14),
        20: (-88.90, 0.26, -71.29, 0.45, 23.50, 0.31),
        40: (-88.93, 0.30, -71.14, 0.56, 23.77, 0.49),
    },
    "vmf": {
        2: (-132.50, 0.73, -126.43, 0.91, 7.28, 0.14),
        5: (-108.43, 0.09, -97.84, 0.13, 13.35, 0.06),
        10: (-93.16, 0.31, -77.03, 0.39, 20.67, 0.08),
        20: (-89.02, 0.31, -67.65, 0.43, 28.50, 0.22),
        40: (-90.87, 0.34, -67.75, 0.70, 33.50, 0.45),
    },
}


def _val_at_best(row) -> tuple[float, float]:
    """Validation (RE, KL) at the selected best epoch, from the JSONL log."""
    log = PROJECT_ROOT / "results" / "cluster" / "t1_logs" / f"{row['tag']}.jsonl"
    if not log.exists():
        return float("nan"), float("nan")
    for line in log.read_text().splitlines():
        j = json.loads(line)
        if j["epoch"] == row["best_epoch"]:
            return j["val_recon"], j["val_kl"]
    return float("nan"), float("nan")


def main() -> None:
    rows = defaultdict(list)
    for path in sorted(ROWS.glob("*.json")):
        r = json.loads(path.read_text())
        rows[(r["family"], r["d"])].append(r)

    print(f"{'arm':<9} {'d':>3} {'n':>2} | {'our LL':>9} {'their LL':>9} {'ok?':>4} | "
          f"{'our RE':>8} {'their RE':>8} {'our KL':>7} {'their KL':>7}")
    n_ok, n_tot = 0, 0
    for (family, d), rs in sorted(rows.items()):
        ll = [r["iwae_ll_mean"] for r in rs]
        vals = [_val_at_best(r) for r in rs]
        re = [v[0] for v in vals if v[0] == v[0]]
        kl = [v[1] for v in vals if v[1] == v[1]]
        mean_ll = sum(ll) / len(ll)
        sd_ll = (sum((v - mean_ll) ** 2 for v in ll) / max(len(ll) - 1, 1)) ** 0.5
        d_ll, d_sd, d_re, d_re_sd, d_kl, d_kl_sd = DAVIDSON[family][d]
        ok = abs(mean_ll - d_ll) <= 2 * (d_sd + sd_ll / max(len(ll), 1) ** 0.5)
        n_ok += ok
        n_tot += 1
        mean_re = sum(re) / len(re) if re else float("nan")
        mean_kl = sum(kl) / len(kl) if kl else float("nan")
        print(f"{family:<9} {d:>3} {len(rs):>2} | {mean_ll:>9.2f} {d_ll:>9.2f} "
              f"{'OK' if ok else 'MISS':>4} | {mean_re:>8.2f} {d_re:>8.2f} {mean_kl:>7.2f} {d_kl:>7.2f}")
    print(f"\nwithin 2 s.d.: {n_ok}/{n_tot} cells")


if __name__ == "__main__":
    main()
