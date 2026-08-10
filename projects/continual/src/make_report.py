"""Render projects/continual/report.html from result CSVs. Self-contained:
inline CSS (repo report tokens) and inline SVG figures. Safe to run with
partial results — sections appear as their data lands.
"""

import datetime
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import analysis as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report.html")

CSS = """
:root { --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
--muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,.10);
--accent:#2a78d6; --s1:#2a78d6; --s2:#1baf7a; --s3:#4a3aa7;
--code-bg:#f2f1ed; --callout-bg:#f2f5fa; }
@media (prefers-color-scheme: dark) { :root { --page:#0d0d0d; --surface:#1a1a19;
--ink:#fff; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
--border:rgba(255,255,255,.10); --accent:#3987e5; --s1:#3987e5; --s2:#199e70;
--s3:#9085e9; --code-bg:#232322; --callout-bg:#1d2430; } }
html,body{background:var(--page)!important;color-scheme:light dark}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink)!important;
line-height:1.6;font-size:15.5px;margin:0}
.wrap{max-width:1060px;margin:0 auto;padding:0 24px 96px;background:var(--page);color:var(--ink)}
.prose{max-width:760px}
header.report-head{padding:56px 0 8px;max-width:760px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
margin:0 0 14px;font-weight:600}
h1{font-size:33px;line-height:1.22;margin:0 0 12px;letter-spacing:-.015em;font-weight:700}
.subtitle{color:var(--ink-2);font-size:17px;margin:0 0 22px;max-width:62ch}
.meta-chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.chip{font-size:12.5px;color:var(--ink-2);border:1px solid var(--border);
border-radius:999px;padding:3px 11px;background:var(--surface)}
h2{font-size:22px;margin:40px 0 10px;letter-spacing:-.01em}
h3{font-size:17px;margin:26px 0 8px}
table{border-collapse:collapse;margin:14px 0;font-size:14px;background:var(--surface)}
th,td{border:1px solid var(--border);padding:6px 12px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:var(--code-bg);font-weight:600}
code{background:var(--code-bg);padding:1px 5px;border-radius:4px;font-size:13.5px}
.callout{background:var(--callout-bg);border:1px solid var(--border);border-radius:10px;
padding:14px 18px;margin:18px 0}
.gate-pass{border-left:4px solid var(--s2)}
.gate-fail{border-left:4px solid #d64541}
.gate-open{border-left:4px solid var(--muted)}
.fig svg{max-width:100%;height:auto}
small{color:var(--muted)}
"""


def chip(t):
    return f'<span class="chip">{t}</span>'


def table(rows, headers):
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    sections = []
    status = []
    changelog = []

    # ---------- Tier 0 ----------
    t0 = A.load_tier("tier0")
    tier0_html = ""
    gate = None
    if len(t0):
        lca = A.run_lca(t0)
        done_runs = lca.run_id.nunique()
        status.append(f"Tier 0: {done_runs}/750 runs")
        expected_pairs = 15
        complete = all(len(lca[(lca.arm == a)]) >= expected_pairs * 25 for a in ("mlp", "nmlp"))
        arms_ready = not (lca[lca.arm == "nmlp"].empty or lca[lca.arm == "mlp"].empty)
        if not arms_ready:
            tier0_html = ("<h2>Tier 0 — replication gate (5+5 CIFAR-10, no rehearsal)</h2>"
                          f"<div class='callout gate-open'>Sweep in progress: {done_runs}/750 runs. "
                          "Both arms needed before any comparison.</div>")
        else:
            cfg_n = A.select_config(lca, "nmlp")
            cfg_m = A.select_config(lca, "mlp")
            gate = A.gate(t0)
            with open(os.path.join(ROOT, "results", "analysis_tier0.json"), "w") as f:
                json.dump(gate, f, indent=2)

            verdict = "PASS" if (gate["pass_gain"] and gate["pass_dominance"]) else "FAIL"
            cls = "gate-pass" if verdict == "PASS" else "gate-fail"
            if not complete:
                verdict, cls = "OPEN (sweep incomplete — provisional numbers below)", "gate-open"
            gate_html = f"""
        <div class="callout {cls}">
        <b>Tier-0 gate: {verdict}</b><br>
        Median paired LCA gain (nMLP − MLP): <b>{fmt(gate['median_gain_pts'])} pts</b>,
        95% bootstrap CI [{fmt(gate['ci95_pts'][0])}, {fmt(gate['ci95_pts'][1])}]
        (n = {gate['n_pairs']} pairs) — criterion ≥ 3.0 pts with CI excluding 0:
        {'met' if gate['pass_gain'] else 'not met'}.<br>
        Frontier dominance on epoch grid: <b>{100*gate['dominance_fraction']:.0f}%</b>
        — criterion ≥ 60%: {'met' if gate['pass_dominance'] else 'not met'}.
            </div>"""
            rows = [
                ["nMLP (selected)", f"{gate['cfg_nmlp']['lr']:g}", f"{gate['cfg_nmlp']['wd']:g}",
                 fmt(100 * gate["cfg_nmlp"]["lca"])],
                ["MLP (selected)", f"{gate['cfg_mlp']['lr']:g}", f"{gate['cfg_mlp']['wd']:g}",
                 fmt(100 * gate["cfg_mlp"]["lca"])],
            ]
            sel_tab = table(rows, ["arm", "lr", "wd", "LCA (%)"])
            lopo = ", ".join(f"{v:+.1f}" for v in gate["lopo_gains_pts"])
            figs = (
                A.fig_epoch_curves(t0, (cfg_n[0], cfg_n[1]), (cfg_m[0], cfg_m[1]))
                + A.fig_frontier(t0, (cfg_n[0], cfg_n[1]), (cfg_m[0], cfg_m[1]))
                + A.fig_heatmaps(lca, A_LR_GRID, A_WD_GRID)
            )
            tier0_html = f"""
        <h2>Tier 0 — replication gate (5+5 CIFAR-10, no rehearsal)</h2>
        {gate_html}
        <p class="prose">Per-arm selected configs (own LR/WD grids; no shared-LR
        comparison):</p>
        {sel_tab}
        <p class="prose"><small>Leave-one-partition-out re-selection gains (pts): {lopo}.
        Selection used the same 15 pairs as evaluation; LOPO is the robustness check.</small></p>
        <div class="fig">{figs}</div>
        """
    else:
        status.append("Tier 0: not started")

    # ---------- Tier 1: WD dose-response ----------
    wd = A.load_tier("wd-dose")
    tier1a_html = ""
    if len(wd) and len(t0):
        lca0 = A.run_lca(t0)
        cfg_m = A.select_config(lca0, "mlp")
        cfg_n = A.select_config(lca0, "nmlp")
        # per-layer update curves: nMLP selected vs baseline doses
        p2 = A.phase2(t0)
        def layer_curve(df, arm, lr, wdv):
            d = df[(df.arm == arm) & (df.lr == lr) & (df.wd == wdv)]
            return np.nanmean(d[["upd0", "upd1", "upd2", "upd3"]].values, axis=0)
        nm_curve = layer_curve(p2, "nmlp", cfg_n[0], cfg_n[1])
        p2w = A.phase2(wd)
        doses = sorted(p2w.wd.unique())
        rows_u, points = [], []
        for d in doses:
            cur = layer_curve(p2w, "mlp", cfg_m[0], d)
            rows_u.append((f"MLP wd={d:g}", "#86b6ef", cur))
            dd = p2w[p2w.wd == d]
            p1w = wd[wd.phase == 1]  # each dose run's own phase-1 old acc
            drops = []
            for rid, grp in dd.groupby("run_id"):
                ref = p1w[p1w.run_id == rid]
                if len(ref):
                    drops.append(ref.old_acc.iloc[0] - grp.old_acc.iloc[-1])
            drops = np.array(drops)
            points.append((d, None, drops.mean() if len(drops) else np.nan,
                           drops.std() / np.sqrt(max(len(drops), 1)) if len(drops) else 0.0))
        rows_u.insert(0, ("nMLP (selected)", "#1baf7a", nm_curve))
        figs = A.fig_update_overlay(rows_u, "Per-layer relative update norms — dose match") + \
               A.fig_dose_response(points, "WD dose–response: old-class forgetting")
        tab = table([[f"{p[0]:g}", fmt(100 * p[2]) + " ± " + fmt(100 * p[3])]
                     for p in points], ["baseline WD", "old-class drop (pts)"])
        tier1a_html = f"""
        <h2>Tier 1a — weight-decay dose–response (ELR control)</h2>
        <p class="prose">Baseline MLP at its selected LR, five WD doses. The dose whose
        per-layer ‖ΔW‖/‖W‖ profile overlays the nMLP's is the ELR-matched point; if the
        nMLP's stability advantage survives there, it is not an effective-LR artifact.</p>
        {tab}
        <div class="fig">{figs}</div>"""
        status.append(f"Tier 1a: {len(wd.run_id.unique())} runs")
    else:
        status.append("Tier 1a (WD dose–response): pending")

    # ---------- Tier 1: LARS ----------
    la = A.load_tier("lars")
    tier1b_html = ""
    if len(la) and len(t0):
        lca_l = A.run_lca(la)
        cfg_l = A.select_config(lca_l.assign(arm="lars"), "lars")  # best LARS config (own grid)
        # frontier of LARS vs tuned AdamW baseline vs nMLP
        lca0 = A.run_lca(t0)
        cfg_m = A.select_config(lca0, "mlp")
        cfg_n = A.select_config(lca0, "nmlp")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
        for df_, arm, cfg, c, lab in [
                (t0, "nmlp", (cfg_n[0], cfg_n[1]), A.C2, "nMLP"),
                (t0, "mlp", (cfg_m[0], cfg_m[1]), A.C1, "MLP (AdamW)"),
                (la, "mlp", (cfg_l[0], cfg_l[1]), A.C3, "MLP (LARS)")]:
            g = A.mean_curves(df_, arm, cfg)
            for ax, col in [(axes[0], "old"), (axes[1], "new")]:
                ax.plot(g.index, 100 * g[col], color=c, label=lab, marker="o", ms=3)
        axes[0].set_ylabel("old-class acc (%)"); axes[1].set_ylabel("new-class acc (%)")
        for ax in axes:
            ax.set_xlabel("phase-2 epoch"); ax.grid(alpha=0.3); ax.legend(frameon=False)
        fig_lars = A._svg(fig)
        lars_lca = A.run_lca(la)
        best = lars_lca[(lars_lca.lr == cfg_l[0]) & (lars_lca.wd == cfg_l[1])].lca.mean()
        tab = table([["MLP + LARS (selected)", f"{cfg_l[0]:g}", f"{cfg_l[1]:g}", fmt(100 * best)],
                     ["nMLP (selected)", f"{cfg_n[0]:g}", f"{cfg_n[1]:g}",
                      fmt(100 * A.select_config(lca0, 'nmlp')[2])],
                     ["MLP + AdamW (selected)", f"{cfg_m[0]:g}", f"{cfg_m[1]:g}",
                      fmt(100 * A.select_config(lca0, 'mlp')[2])]],
                    ["arm", "lr", "wd", "LCA (%)"])
        tier1b_html = f"""
        <h2>Tier 1b — LARS reference arm (scale-invariant optimizer)</h2>
        <p class="prose">If normalization's benefit were only scale-invariance of the
        effective step, LARS on the plain MLP should reproduce the nMLP frontier. LARS is
        swept on its own LR grid.</p>
        {tab}
        <div class="fig">{fig_lars}</div>"""
        status.append(f"Tier 1b: {len(la.run_id.unique())} runs")
    else:
        status.append("Tier 1b (LARS): pending")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    changelog.append(f"{now} — report regenerated ({'; '.join(status)})")
    chlog_path = os.path.join(ROOT, "results", "changelog.txt")
    old_chlog = ""
    if os.path.exists(chlog_path):
        old_chlog = open(chlog_path).read()
    with open(chlog_path, "w") as f:
        f.write((changelog[0] + "\n" + old_chlog).strip() + "\n")
    chlog_html = "<br>".join((changelog[0] + "\n" + old_chlog).strip().split("\n")[:15])

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project 1 — Normalized networks in class-incremental learning</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<header class="report-head">
<p class="eyebrow">Research program on hyperspherical representations · Project 1</p>
<h1>Normalized networks in class-incremental learning</h1>
<p class="subtitle">Does nGPT-style hyperspherical normalization improve the
stability–plasticity frontier in rehearsal-free class-incremental learning —
and is bounded angular motion of representations the reason?</p>
<div class="meta-chips">
{chip('substrate: CIFAR-10 5+5, no rehearsal')}
{chip('arms: tuned MLP vs nGPT-style nMLP')}
{chip('per-arm LR×WD sweeps; no shared LR')}
{chip('5 partitions × 3 seeds, CRN-paired')}
{chip('pre-registered gate: commit b419d17')}
{chip('device: Apple M3 Max (MPS), fp32')}
</div></header>

<h2>Status</h2>
<div class="callout gate-open"><b>{" · ".join(status)}</b><br>
<small>Frontier, not endpoints, is the comparison object. Gate: median paired
LCA gain ≥ 3.0 pts (CI excl. 0) AND ≥ 60% epoch-grid dominance.</small></div>

{tier0_html}
{tier1a_html}
{tier1b_html}

<h2>Setup &amp; protocol notes</h2>
<p class="prose">Phase 1: 15 epochs on 5 classes; head extended by 5 rows;
phase 2: 15 epochs on the other 5, no rehearsal. Old/new accuracy use argmax
over all 10 logits (task-agnostic CIL). No augmentation; inputs normalized.
Optimizer AdamW unless stated; fresh optimizer state per phase. The nMLP uses
functionally normalized rows (scale-invariant weights) with learnable per-row
eigen-learning-rates, unit-norm activations after each hidden layer, and a
cosine classifier with learnable logit scale — weight decay is therefore its
effective-LR knob, which is exactly what Tier 1a exploits for the control.
Negative results and failed gates are reported as-is.</p>

<h2>Changelog</h2>
<p class="prose"><small>{chlog_html}</small></p>
</div></body></html>"""
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT} ({len(html)/1024:.0f} KB)")


A_LR_GRID = {"mlp": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
             "nmlp": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]}
A_WD_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

if __name__ == "__main__":
    main()
