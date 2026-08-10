"""Generate task lists (one line per run_t2.py invocation).

Stages (see PREREGISTRATION.md):
  t1     — Davidson reproduction anchor: d in {2,5,10,20,40} x {gaussian, vmf}
           x 5 seeds at his exact settings (lr 1e-3, batch 64, warmup 100).
  pilot  — per-arm LR sweep, LR in {3e-4,1e-3,3e-3} at anchor dims
           d in {20,128,1024}, 1 seed, 4 arms.  Selects best LR per (arm,
           anchor d); final runs use the nearest anchor's LR (log scale).
  final  — T2 grid: d in {5..1024} x 4 arms x 5 seeds at the selected LRs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

D_GRID = [5, 10, 20, 40, 64, 128, 256, 512, 1024]
ARMS = ["gaussian", "gaussian_tied", "vmf", "power_spherical"]
PILOT_D = [20, 128, 1024]
LR_GRID = [3e-4, 1e-3, 3e-3]
SEEDS = [0, 1, 2, 3, 4]


def line(family, d, lr, seed, epochs, out, logdir, dataset="mnist", extra=""):
    return (
        f"--dataset {dataset} --family {family} --d {d} --lr {lr:g} --seed {seed} "
        f"--epochs {epochs} --out {out} --log-dir {logdir} {extra}".strip()
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["t1", "pilot", "final"], required=True)
    args = p.parse_args()

    lines = []
    if args.stage == "t1":
        for family in ["gaussian", "vmf"]:
            for d in [2, 5, 10, 20, 40]:
                for seed in SEEDS:
                    lines.append(line(family, d, 1e-3, seed, 500, "results/t1_rows", "results/t1_logs"))
    elif args.stage == "pilot":
        for family in ARMS:
            for d in PILOT_D:
                for lr in LR_GRID:
                    lines.append(
                        line(family, d, lr, 0, 300, "results/t2_pilot_rows", "results/t2_pilot_logs")
                    )
    else:
        rows_dir = PROJECT_ROOT / "results" / "t2_pilot_rows"
        best_ll: dict[tuple, float] = {}
        for path in sorted(rows_dir.glob("*.json")):
            r = json.loads(path.read_text())
            key = (r["family"], r["d"], r["lr"])
            best_ll[key] = max(best_ll.get(key, -1e30), r["iwae_ll_mean"])
        anchors: dict[str, dict[int, float]] = {}
        for family in ARMS:
            anchors[family] = {}
            for d in PILOT_D:
                anchors[family][d] = max(
                    (lr for lr in LR_GRID), key=lambda lr: best_ll.get((family, d, lr), -1e30)
                )
        for family in ARMS:
            for d in D_GRID:
                # nearest anchor in log-d
                anchor = min(PILOT_D, key=lambda a: abs(math.log(a / d)))
                lr = anchors[family][anchor]
                for seed in SEEDS:
                    lines.append(
                        line(family, d, lr, seed, 500, "results/t2_rows", "results/t2_logs")
                    )
        print("selected LRs:", json.dumps({f: {str(d): lr for d, lr in a.items()} for f, a in anchors.items()}))

    out = PROJECT_ROOT / "slurm" / f"t2_tasks_{args.stage}.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} tasks -> {out}")
    print("submit: sbatch --array=0-%d%%2 --export=ALL,TASKS_FILE=.../%s projects/hyperspherical-vae/slurm/t2_array.slurm" % (len(lines) - 1, out.name))


if __name__ == "__main__":
    main()
