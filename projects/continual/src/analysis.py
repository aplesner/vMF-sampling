"""Analysis for Project 1 Tier 0/1: frontier metrics, gate statistics,
and figures (inline SVG strings for the report).
"""

import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

N_BOOT = 10_000
RNG = np.random.default_rng(7)


def load_tier(tier, results_dir=None):
    frames = []
    rdir = results_dir or RESULTS
    for f in sorted(glob.glob(os.path.join(rdir, f"{tier}_shard*.csv"))):
        frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["run_id", "phase", "epoch"], keep="first")


def phase2(df):
    return df[df.phase == 2].copy()


def run_lca(df):
    """Learning-curve area per run: mean over phase-2 epochs of avg_acc."""
    p2 = phase2(df)
    return (p2.groupby("run_id")
              .agg(lca=("avg_acc", "mean"),
                   arm=("arm", "first"), opt=("opt", "first"),
                   lr=("lr", "first"), wd=("wd", "first"),
                   partition=("partition", "first"), seed=("seed", "first"),
                   final_old=("old_acc", "last"), final_new=("new_acc", "last"))
              .reset_index())


def select_config(lca, arm, exclude_partition=None):
    d = lca[lca.arm == arm]
    if exclude_partition is not None:
        d = d[d.partition != exclude_partition]
    g = d.groupby(["lr", "wd"]).lca.mean()
    if len(g) == 0:
        return None
    lr, wd = g.idxmax()
    return lr, wd, g.max()


def paired_gains(lca, cfg_a, cfg_b):
    """Paired by (partition, seed): arm_a gain minus arm_b, each at its own config."""
    out = []
    for (p, s), grp in lca.groupby(["partition", "seed"]):
        a = grp[(grp.lr == cfg_a[0]) & (grp.wd == cfg_a[1]) & (grp.arm == "nmlp")]
        b = grp[(grp.lr == cfg_b[0]) & (grp.wd == cfg_b[1]) & (grp.arm == "mlp")]
        if len(a) and len(b):
            out.append(float(a.lca.iloc[0] - b.lca.iloc[0]))
    return np.array(out)


def boot_median_ci(x, n_boot=N_BOOT, alpha=0.05):
    meds = [np.median(RNG.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def mean_curves(df, arm, cfg):
    """Mean phase-2 epoch curves (over pairs) at a config."""
    d = phase2(df)
    d = d[(d.arm == arm) & (d.lr == cfg[0]) & (d.wd == cfg[1])]
    g = d.groupby("epoch").agg(old=("old_acc", "mean"), new=("new_acc", "mean"),
                               old_se=("old_acc", lambda s: s.std() / np.sqrt(len(s))),
                               new_se=("new_acc", lambda s: s.std() / np.sqrt(len(s))))
    return g.sort_index()


def dominance(df, cfg_nmlp, cfg_mlp):
    a = mean_curves(df, "nmlp", cfg_nmlp)
    b = mean_curves(df, "mlp", cfg_mlp)
    dom = (a.old.values >= b.old.values) & (a.new.values >= b.new.values)
    return float(dom.mean()), dom


def gate(df):
    lca = run_lca(df)
    cfg_n = select_config(lca, "nmlp")
    cfg_m = select_config(lca, "mlp")
    gains = paired_gains(lca, cfg_n, cfg_m)
    med = float(np.median(gains))
    lo, hi = boot_median_ci(gains)
    frac, dom = dominance(df, cfg_n, cfg_m)
    # leave-one-partition-out robustness
    lopo = []
    for p in sorted(lca.partition.unique()):
        cn = select_config(lca, "nmlp", exclude_partition=p)
        cm = select_config(lca, "mlp", exclude_partition=p)
        if cn is None or cm is None:
            continue
        sub = lca[lca.partition == p]
        g = paired_gains(sub, cn, cm)
        if len(g):
            lopo.append(float(g[0]))
    return dict(
        cfg_nmlp=dict(lr=cfg_n[0], wd=cfg_n[1], lca=cfg_n[2]),
        cfg_mlp=dict(lr=cfg_m[0], wd=cfg_m[1], lca=cfg_m[2]),
        n_pairs=int(len(gains)),
        median_gain_pts=med * 100,
        ci95_pts=[lo * 100, hi * 100],
        dominance_fraction=frac,
        dominance_by_epoch=dom.tolist(),
        lopo_gains_pts=[v * 100 for v in lopo],
        pass_gain=bool(med * 100 >= 3.0 and lo > 0),
        pass_dominance=bool(frac >= 0.60),
    )


def pareto_frontier(points):
    """points: (N,2) array of (new, old). Returns sorted non-dominated set."""
    pts = np.asarray(points)
    keep = []
    for p in pts:
        dominated = ((pts[:, 0] >= p[0]) & (pts[:, 1] >= p[1])
                     & ((pts[:, 0] > p[0]) | (pts[:, 1] > p[1]))).any()
        if not dominated:
            keep.append(p)
    return np.array(sorted(keep, key=lambda x: x[0]))


def pooled_frontier(df, arm):
    """Pareto frontier across all (config, epoch) mean (new, old) points."""
    d = phase2(df)
    d = d[d.arm == arm]
    g = d.groupby(["lr", "wd", "epoch"]).agg(old=("old_acc", "mean"),
                                              new=("new_acc", "mean")).reset_index()
    front = pareto_frontier(g[["new", "old"]].values)
    auc = float(np.trapezoid(front[:, 1], front[:, 0])) if len(front) > 1 else 0.0
    return front, auc


def fig_pooled_frontier(fronts_aucs, title="Pooled stability–plasticity frontier (all configs × epochs)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for lab, c, front, auc in fronts_aucs:
        ax.plot(100 * front[:, 0], 100 * front[:, 1], marker="o", ms=3.5,
                color=c, label=f"{lab} (AUC {auc:.3f})")
    ax.set_xlabel("plasticity: new-class acc (%)")
    ax.set_ylabel("stability: old-class acc (%)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(title, fontsize=10)
    return _svg(fig)


# ---------------- figures (inline SVG) ----------------

def _svg(fig):
    import io
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return buf.getvalue().split("<svg", 1)[1].join(["<svg", ""]) if False else \
        "<svg" + buf.getvalue().split("<svg", 1)[1]


C1, C2, C3 = "#2a78d6", "#1baf7a", "#4a3aa7"


def fig_epoch_curves(df, cfg_nmlp, cfg_mlp, title="Phase 2: stability & plasticity vs epoch"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for arm, cfg, c, lab in [("nmlp", cfg_nmlp, C2, "nMLP"), ("mlp", cfg_mlp, C1, "MLP")]:
        g = mean_curves(df, arm, cfg)
        for ax, col, se in [(axes[0], "old", "old_se"), (axes[1], "new", "new_se")]:
            ax.plot(g.index, 100 * g[col], color=c, label=lab, marker="o", ms=3)
            ax.fill_between(g.index, 100 * (g[col] - g[se]), 100 * (g[col] + g[se]),
                            color=c, alpha=0.15, lw=0)
    axes[0].set_ylabel("old-class acc (%)")
    axes[1].set_ylabel("new-class acc (%)")
    for ax in axes:
        ax.set_xlabel("phase-2 epoch")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle(title, y=1.02, fontsize=11)
    return _svg(fig)


def fig_frontier(df, cfg_nmlp, cfg_mlp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for arm, cfg, c, lab in [("nmlp", cfg_nmlp, C2, "nMLP"), ("mlp", cfg_mlp, C1, "MLP")]:
        g = mean_curves(df, arm, cfg)
        ax.plot(100 * g["new"], 100 * g["old"], color=c, marker="o", ms=3, label=lab)
        ax.annotate("ep1", (100 * g["new"].iloc[0], 100 * g["old"].iloc[0]),
                    fontsize=8, color=c)
    ax.set_xlabel("plasticity: new-class acc (%)")
    ax.set_ylabel("stability: old-class acc (%)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Stability–plasticity trajectory over phase-2 epochs", fontsize=10)
    return _svg(fig)


def fig_heatmaps(lca, lr_grids, wds):
    """lr_grids: dict arm -> list of LRs (per-arm grids; no shared LR)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, arm, lab in [(axes[0], "mlp", "MLP"), (axes[1], "nmlp", "nMLP")]:
        lrs = lr_grids[arm]
        d = lca[lca.arm == arm].groupby(["lr", "wd"]).lca.mean() * 100
        if len(d) == 0:
            continue
        m = np.full((len(wds), len(lrs)), np.nan)
        for i, wd in enumerate(wds):
            for j, lr in enumerate(lrs):
                if (lr, wd) in d.index:
                    m[i, j] = d.loc[(lr, wd)]
        im = ax.imshow(m, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(lrs)), [f"{v:g}" for v in lrs], fontsize=8)
        ax.set_yticks(range(len(wds)), [f"{v:g}" for v in wds], fontsize=8)
        ax.set_xlabel("learning rate")
        ax.set_ylabel("weight decay")
        ax.set_title(f"{lab} — LCA (%)", fontsize=10)
        for i in range(len(wds)):
            for j in range(len(lrs)):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i,j]:.1f}", ha="center", va="center",
                            fontsize=7, color="white" if m[i, j] < np.nanmax(m) - 8 else "black")
        fig.colorbar(im, ax=ax, shrink=0.85)
    return _svg(fig)


def fig_update_overlay(rows, title):
    """rows: list of (label, color, per-layer mean rel-update arrays [n_layers])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for lab, c, arr in rows:
        ax.plot(range(len(arr)), arr, marker="o", ms=4, color=c, label=lab)
    ax.set_xlabel("layer (0=input … last=head)")
    ax.set_ylabel("mean per-step ||ΔW||/||W|| (phase 2)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    return _svg(fig)


def fig_dose_response(points, title):
    """points: list of (wd, matched rel-upd, old-drop mean, old-drop se, lca mean)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wds = [p[0] for p in points]
    drop = [100 * p[2] for p in points]
    se = [100 * p[3] for p in points]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(range(len(wds)), drop, yerr=se, marker="o", color=C3, capsize=3)
    ax.set_xticks(range(len(wds)), [f"{v:g}" for v in wds], fontsize=8)
    ax.set_xlabel("baseline weight decay (dose)")
    ax.set_ylabel("old-class accuracy drop, end of phase 2 (pts)")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    return _svg(fig)
