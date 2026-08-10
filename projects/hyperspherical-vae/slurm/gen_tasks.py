"""Generate the T2 task list (one line per run_t2.py invocation).

Protocol: per-(arm, d) LR sweep with 2 pilot seeds first; the full 5-seed
runs then use the best LR per (arm, d).  This script emits both stages; the
pilot stage is submitted first, selection happens analysis-side, then the
final stage is generated with --stage final (reads pilot rows).

Grid (brief T2): d in {5,10,20,40,64,128,256,512,1024}, arms
{gaussian, gaussian_tied, vmf, power_spherical}, MNIST.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

D_GRID = [5, 10, 20, 40, 64, 128, 256, 512, 1024]
ARMS = ["gaussian", "gaussian_tied", "vmf", "power_spherical"]
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3]
PILOT_SEEDS = [0, 1]
FINAL_SEEDS = [0, 1, 2, 3, 4]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pilot", "final"], required=True)
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--epochs", type=int, default=500)
    args = p.parse_args()

    lines = []
    if args.stage == "pilot":
        for family in ARMS:
            for d in D_GRID:
                for lr in LR_GRID:
                    for seed in PILOT_SEEDS:
                        lines.append(
                            f"--dataset {args.dataset} --family {family} --d {d} "
                            f"--lr {lr:g} --seed {seed} --epochs {args.epochs} "
                            f"--out results/t2_pilot_rows --log-dir results/t2_pilot_logs"
                        )
    else:
        rows_dir = PROJECT_ROOT / "results" / "t2_pilot_rows"
        best: dict[tuple[str, int], float] = {}
        for path in sorted(rows_dir.glob("*.json")):
            r = json.loads(path.read_text())
            key = (r["family"], r["d"], r["lr"])
            best[key] = max(best.get(key, -1e30), r["iwae_ll_mean"])
        best_lr: dict[tuple[str, int], float] = {}
        for (family, d, lr), ll in best.items():
            k = (family, d)
            if k not in best_lr or ll > best[(family, d, best_lr[k])]:
                best_lr[k] = lr
        for (family, d), lr in sorted(best_lr.items()):
            for seed in FINAL_SEEDS:
                lines.append(
                    f"--dataset {args.dataset} --family {family} --d {d} "
                    f"--lr {lr:g} --seed {seed} --epochs {args.epochs} "
                    f"--out results/t2_rows --log-dir results/t2_logs"
                )

    out = PROJECT_ROOT / "slurm" / f"t2_tasks_{args.stage}.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} tasks -> {out}")
    print("submit: sbatch --array=0-%d%%2 projects/hyperspherical-vae/slurm/t2_array.slurm" % (len(lines) - 1))


if __name__ == "__main__":
    main()
