"""Aggregate a sweep CSV into band-width tables (the headline result) and
the mandatory effective-step re-expression.

Usage: python src/analyze.py results/<run>/sweeps.csv
"""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd

from bandwidth import (RunRecord, band_width, band_width_rel,
                       reexpress_bands_effstep)


def load(path: str) -> list[RunRecord]:
    df = pd.read_csv(path)
    return [
        RunRecord(arm=r["arm"], rank=int(r["rank"]), lr=float(r["lr"]),
                  seed=int(r["seed"]), metric=float(r["metric"]),
                  nan_encountered=bool(r["nan_encountered"]),
                  loss_init=float(r["loss_init"]),
                  loss_final_window=float(r["loss_final_window"]),
                  grad_norm_max=float(r["grad_norm_max"]),
                  eff_step_run_mean=float(r["eff_step_run_mean"]))
        for r in df.to_dict("records")
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--tau-abs", type=float, default=1.0)
    p.add_argument("--tau-rel", type=float, default=0.02)
    args = p.parse_args()

    recs = load(args.csv)
    arms = sorted({r.arm for r in recs})
    ranks = sorted({r.rank for r in recs})

    print(f"\nBand width in decades (tau = {args.tau_abs} accuracy point abs | "
          f"{args.tau_rel:.0%} rel), seed-mean over final val metric\n")
    hdr = f"{'arm':<12}" + "".join(f"r={r:<4}" for r in ranks) + " eta_div"
    print(hdr + "\n" + "-" * len(hdr))
    bands = {}
    for arm in arms:
        row = f"{arm:<12}"
        for rank in ranks:
            sub = [r for r in recs if r.arm == arm and r.rank == rank]
            if not sub:
                row += f"{'—':<6}"
                continue
            b = band_width(sub, args.tau_abs, args.tau_rel, arm, rank)
            bands[(arm, rank)] = b
            w = b.width_decades
            row += f"{w if not math.isnan(w) else 'div':<6.2f}"
        divs = [b.eta_div for (a, _), b in bands.items() if a == arm]
        divs = [d for d in divs if not math.isnan(d)]
        row += f" {min(divs):.1e}" if divs else "   —"
        print(row)

    print("\nRelative-tolerance widths (tau = {:.0%} of own peak):".format(args.tau_rel))
    for (arm, rank), b in sorted(bands.items()):
        sub = [r for r in recs if r.arm == arm and r.rank == rank]
        w = band_width_rel(sub, args.tau_rel)
        print(f"  {arm:<12} r={rank:<3} W_rel = "
              f"{w if not math.isnan(w) else float('nan'):.2f} decades")

    print("\nPre-registered contrasts (tau_abs):")
    for rank in ranks:
        if (rank) and ("delora", rank) in bands and ("lora", rank) in bands:
            norm = bands[("delora", rank)].width_decades - bands[("lora", rank)].width_decades
            print(f"  r={rank:<3} normalization  W(DeLoRA)-W(LoRA)      = {norm:+.2f} dec")
        if ("delora-pc", rank) in bands and ("delora", rank) in bands:
            pc = bands[("delora-pc", rank)].width_decades - bands[("delora", rank)].width_decades
            tag = "  [r=1 exact null: must be 0]" if rank == 1 else ""
            print(f"  r={rank:<3} per-component W(DeLoRA-PC)-W(DeLoRA) = {pc:+.2f} dec{tag}")

    print("\nFalsifier 1 — bands re-expressed in effective-step units "
          "(must NOT coincide; if they do, the widening is an artifact):")
    for rank in ranks:
        sub = [r for r in recs if r.rank == rank]
        widths = reexpress_bands_effstep(sub, args.tau_abs)
        if widths:
            s = "  ".join(f"{a}={w:.2f}" for a, w in sorted(widths.items())
                          if not math.isnan(w))
            print(f"  r={rank:<3} {s}")


if __name__ == "__main__":
    sys.exit(main())
