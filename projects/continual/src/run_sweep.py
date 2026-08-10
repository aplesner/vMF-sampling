"""Sweep driver for Project 1 tiers. Shardable; appends to per-shard CSVs;
skips run_ids already present (resume-safe).

Usage:
  python run_sweep.py --tier tier0 --shard 0 --nshards 4
  python run_sweep.py --tier wd-dose --wd-points 0,3e-4,1e-3,3e-3,1e-2 --lr 1e-3
  python run_sweep.py --tier lars
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from data import make_partitions
from engine import run_config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

# Per-arm LR grids: arms are never compared at a shared LR; each grid
# brackets its arm's useful range (MLP collapses at >=1e-4-ish on the
# stability axis; the scale-invariant nMLP needs 100x larger steps).
LR_GRID = {"mlp": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
           "nmlp": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]}
WD_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
LARS_LR_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
LARS_WD_GRID = [0.0, 1e-3]
N_PARTITIONS = 5
N_SEEDS = 3

FIELDS = ["run_id", "tier", "arm", "opt", "lr", "wd", "partition", "seed",
          "phase", "epoch", "old_acc", "new_acc", "avg_acc",
          "upd0", "upd1", "upd2", "upd3", "wall_s"]


def build_grid(tier, args):
    runs = []
    if tier == "tier0":
        for arm in ("mlp", "nmlp"):
            for lr in LR_GRID[arm]:
                for wd in WD_GRID:
                    runs.append(dict(arm=arm, opt_name="adamw", lr=lr, wd=wd))
    elif tier == "mlp-lowlr":
        # ELR control: does the plain MLP show the nMLP's slow-decay
        # transient once its steps are small enough?
        for lr in (1e-6, 3e-6):
            for wd in (0.0, 1e-3):
                runs.append(dict(arm="mlp", opt_name="adamw", lr=lr, wd=wd))
    elif tier == "wd-dose":
        for wd in [float(x) for x in args.wd_points.split(",")]:
            runs.append(dict(arm="mlp", opt_name="adamw", lr=args.lr, wd=wd))
    elif tier == "lars":
        for lr in LARS_LR_GRID:
            for wd in LARS_WD_GRID:
                runs.append(dict(arm="mlp", opt_name="lars", lr=lr, wd=wd))
    else:
        raise ValueError(tier)
    full = []
    for cfg in runs:
        for p in range(N_PARTITIONS):
            for s in range(N_SEEDS):
                full.append(dict(cfg, partition=p, seed=2000 + s))
    if tier == "tier0":
        # interleave arms so both make progress together (round-robin sharding)
        mlp_runs = [c for c in full if c["arm"] == "mlp"]
        nmlp_runs = [c for c in full if c["arm"] == "nmlp"]
        full = [c for pair in zip(mlp_runs, nmlp_runs) for c in pair]
    return full


def run_id(tier, cfg):
    return f"{tier}:{cfg['arm']}:{cfg['opt_name']}:{cfg['lr']}:{cfg['wd']}:{cfg['partition']}:{cfg['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--epochs-p1", type=int, default=15)
    ap.add_argument("--epochs-p2", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd-points", type=str, default="")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    grid = build_grid(args.tier, args)
    if args.smoke:
        grid = grid[:1]
        args.epochs_p1 = args.epochs_p2 = 2
    grid = [c for i, c in enumerate(grid) if i % args.nshards == args.shard]

    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, f"{args.tier}_shard{args.shard}.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {r["run_id"] for r in csv.DictReader(f)}
    new_file = not os.path.exists(out)

    partitions = make_partitions(N_PARTITIONS)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for cfg in grid:
            rid = run_id(args.tier, cfg)
            if rid in done:
                continue
            t0 = time.time()
            rows = run_config(**cfg, partitions=partitions,
                              data_dir=args.data_dir, device=args.device,
                              epochs_p1=args.epochs_p1, epochs_p2=args.epochs_p2,
                              batch_size=args.batch_size, augment=True)
            wall = time.time() - t0
            for r in rows:
                w.writerow(dict(run_id=rid, tier=args.tier, arm=cfg["arm"],
                                opt=cfg["opt_name"], lr=cfg["lr"], wd=cfg["wd"],
                                partition=cfg["partition"], seed=cfg["seed"],
                                wall_s=f"{wall:.1f}", **r))
            f.flush()
            print(f"[{args.tier} shard{args.shard}] {rid} {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
