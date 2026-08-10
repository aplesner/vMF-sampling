"""Render projects/kappa-measure/report.html from the results CSVs.

Self-contained output: inline CSS (design tokens inherited from
docs/kappa_report.html), inline SVG, no network, no build step.  Lead with
status; every claim ships its table.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
RESULTS = PROJECT_ROOT / "results"

PRE_REGISTRATION_COMMIT = "2f146ef"


def repo_css() -> str:
    """Extract the <style> block of docs/kappa_report.html to inherit tokens."""

    html = (REPO_ROOT / "docs" / "kappa_report.html").read_text()
    match = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
    if not match:
        raise RuntimeError("could not extract style block from docs/kappa_report.html")
    return match.group(1)


def git_hash() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()


def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def svg_line_plot(xs, ys, *, xlabel, ylabel, hline=None, width=640, height=300):
    """Minimal inline-SVG line plot on log-x scale."""

    pad_l, pad_r, pad_t, pad_b = 64, 20, 18, 42
    lx = np.log10(np.asarray(xs, dtype=float))
    x0, x1 = lx.min(), lx.max()
    y0 = 0.0
    y1 = max(max(ys) * 1.08, (hline or 0) * 1.08)

    def X(v):
        return pad_l + (np.log10(v) - x0) / (x1 - x0) * (width - pad_l - pad_r)

    def Y(v):
        return pad_t + (1 - (v - y0) / (y1 - y0)) * (height - pad_t - pad_b)

    points = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{ylabel} vs {xlabel}">',
        f'<line x1="{pad_l}" y1="{Y(y0):.1f}" x2="{width - pad_r}" y2="{Y(y0):.1f}" stroke="var(--baseline)"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{Y(y0):.1f}" stroke="var(--baseline)"/>',
    ]
    for gx in xs:
        parts.append(
            f'<text x="{X(gx):.1f}" y="{height - pad_b + 16:.1f}" font-size="10.5" '
            f'fill="var(--muted)" text-anchor="middle">{gx:g}</text>'
        )
    for gy in (0.0, 1.0, 2.0, 3.0):
        if y0 <= gy <= y1:
            parts.append(
                f'<text x="{pad_l - 8}" y="{Y(gy) + 3.5:.1f}" font-size="10.5" '
                f'fill="var(--muted)" text-anchor="end">{gy:g}</text>'
                f'<line x1="{pad_l}" y1="{Y(gy):.1f}" x2="{width - pad_r}" y2="{Y(gy):.1f}" '
                f'stroke="var(--grid)" stroke-width="0.7"/>'
            )
    if hline is not None:
        parts.append(
            f'<line x1="{pad_l}" y1="{Y(hline):.1f}" x2="{width - pad_r}" y2="{Y(hline):.1f}" '
            f'stroke="var(--s2)" stroke-dasharray="5 4" stroke-width="1.4"/>'
            f'<text x="{width - pad_r - 4}" y="{Y(hline) - 5:.1f}" font-size="10.5" '
            f'fill="var(--s2)" text-anchor="end">√(n_B/n_A) = {hline:.2f}</text>'
        )
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="var(--accent)" stroke-width="2"/>'
    )
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="var(--accent)"/>')
    parts.append(
        f'<text x="{(pad_l + width - pad_r) / 2:.1f}" y="{height - 6}" font-size="11.5" '
        f'fill="var(--ink-2)" text-anchor="middle">{xlabel}</text>'
    )
    parts.append(
        f'<text x="14" y="{(pad_t + height - pad_b) / 2:.1f}" font-size="11.5" fill="var(--ink-2)" '
        f'text-anchor="middle" transform="rotate(-90 14 {(pad_t + height - pad_b) / 2:.1f})">{ylabel}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    surface = pd.read_csv(RESULTS / "surface_validation.csv")
    audit = pd.read_csv(RESULTS / "audit_verdicts.csv")
    nulls = pd.read_csv(RESULTS / "null_quantiles.csv")

    ratio = surface["ratio_emp_over_pred"]
    sweep = audit[audit["claim_id"].str.startswith("sweep-kop")].copy()
    sweep["kop"] = sweep["claim_id"].str.replace("sweep-kop-", "", regex=False).astype(float)
    sweep = sweep.sort_values("kop")
    canonical = audit[audit["claim_id"] == "canonical-unequal-n"].iloc[0]
    illustration = audit[audit["claim_id"] == "brief-illustration-n100"].iloc[0]
    control = audit[audit["claim_id"] == "control-matched-n"].iloc[0]
    true2 = audit[audit["claim_id"] == "control-true-ratio-2"].iloc[0]

    calib = surface[
        (surface["p"] == 512)
        & (np.isclose(surface["kappa_over_p"], 3.0))
        & (np.isclose(surface["n_over_p"], 0.5))
    ].iloc[0]

    # Surface validation table, condensed: worst |ratio-1| per (kappa/p, n/p).
    pivot = (
        surface.assign(dev=(surface["ratio_emp_over_pred"] - 1).abs())
        .groupby(["kappa_over_p", "n_over_p"])["ratio_emp_over_pred"]
        .agg(["min", "max"])
        .reset_index()
    )

    def cell_class(lo: float, hi: float) -> str:
        if lo >= 0.5 and hi <= 2.0:
            return "cell-ok"
        return "cell-warn"

    surf_rows = "\n".join(
        "<tr>"
        f'<td class="rowhead">κ/p = {r.kappa_over_p:g}</td>'
        f'<td class="rowhead">n/p = {r.n_over_p:g}</td>'
        f'<td class="num {cell_class(r.min, r.max)}">{r.min:.3f} – {r.max:.3f}</td>'
        "</tr>"
        for r in pivot.itertuples()
    )

    audit_rows = "\n".join(
        "<tr>"
        f'<td class="rowhead">{r.claim_id}</td>'
        f'<td class="num">{r.kappa_over_p:g}</td>'
        f'<td class="num">{r.n_a} / {r.n_b}</td>'
        f'<td class="num">{r.ratio_true:.2f}</td>'
        f'<td class="num"><b>{r.ratio_observed:.3f}</b></td>'
        f'<td class="{"cell-warn" if "not resolvable" in r.verdict else "cell-ok"}">{r.verdict}</td>'
        "</tr>"
        for r in audit.itertuples()
    )

    kop_uniform = nulls[
        (nulls["statistic"] == "r_bar") & (nulls["null"] == "uniform")
        & (nulls["p"] == 512) & (nulls["n"] == 70) & (nulls["quantile"] == 0.95)
    ].iloc[0]["value"]
    mr = nulls[
        (nulls["statistic"] == "r_bar") & (nulls["null"] == "mean_removed")
        & (nulls["p"] == 512) & (nulls["n"] == 70) & (nulls["quantile"] == 0.95)
    ].iloc[0]["value"]

    plot = svg_line_plot(
        list(sweep["kop"]),
        list(sweep["ratio_observed"]),
        xlabel="κ/p (log scale)",
        ylabel="spurious κ̂ ratio, n_A=70 vs n_B=700",
        hline=float(np.sqrt(700 / 70)),
    )

    today = date.today().isoformat()
    head_hash = git_hash()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project 6 — Measuring representation geometry: nulls, feasibility, refusal</title>
<style>{repo_css()}</style>
</head>
<body>
<div class="wrap">
<header class="report-head">
  <p class="eyebrow">Instrument &amp; calibration report · Project 6 · vMF-sampling</p>
  <h1>Measuring representation geometry: nulls, feasibility, and an estimator that refuses</h1>
  <p class="subtitle">
    <code>vmf-measure</code> v0.1.0: matched null quantiles, the closed-form admissibility
    surface that retires the rectangular rule, a log-Bessel path valid at p ≥ 8192, and a
    refusal mode that returns “not resolvable at this n” instead of a number.
  </p>
  <div class="meta-chips">
    <span class="chip">Schema &amp; API <b>frozen · v0.1.0</b></span>
    <span class="chip">Owner <b>A. Plesner</b></span>
    <span class="chip">Pre-registration <b>{PRE_REGISTRATION_COMMIT}</b></span>
    <span class="chip">Apple <b>M3 Max</b> CPU, float64</span>
    <span class="chip">Updated <b>{today}</b></span>
  </div>
</header>

<div class="tiles">
  <div class="tile">
    <p class="label">API &amp; table schema</p>
    <p class="value">frozen</p>
    <p class="note"><code>SCHEMA.md</code> pins signatures, enums, CSV columns, and the refusal string. Projects 1 and 5 pin the schema, not the numbers.</p>
  </div>
  <div class="tile">
    <p class="label">Surface vs Monte Carlo</p>
    <p class="value">{ratio.min():.2f}–{ratio.max():.2f}</p>
    <p class="note">emp/pred ratio across the {len(surface)}-cell grid; acceptance target was a factor of 2. The closed form is exact in resolvable cells and conservative only deep in the refusal region.</p>
  </div>
  <div class="tile">
    <p class="label">Spurious κ̂ ratio, identical true κ</p>
    <p class="value">{canonical["ratio_observed"]:.2f}×</p>
    <p class="note">n_A=70 vs n_B=700 at κ/p=0.05. Rises to {sweep["ratio_observed"].max():.2f}× near the uniform null; bounded by √(n_B/n_A).</p>
  </div>
  <div class="tile">
    <p class="label">log-Bessel at p = 8192–32768</p>
    <p class="value">2e-16</p>
    <p class="note">Relative error against a 50-digit mpmath reference at orders 4095–16383, where SciPy's <code>ive</code> returns exactly 0.</p>
  </div>
</div>

<nav class="toc" aria-label="Contents">
  <ol>
    <li><span class="n">1</span><a href="#status">Status</a></li>
    <li><span class="n">2</span><a href="#tool">The tool</a></li>
    <li><span class="n">3</span><a href="#bessel">Log-Bessel at p ≥ 8192</a></li>
    <li><span class="n">4</span><a href="#surface">D1: surface validation</a></li>
    <li><span class="n">5</span><a href="#audit">Audit: unequal n, low κ/p</a></li>
    <li><span class="n">6</span><a href="#nulls">Nulls</a></li>
    <li><span class="n">7</span><a href="#card">Measurement card (D4)</a></li>
    <li><span class="n">8</span><a href="#scope">Scope &amp; caveats</a></li>
    <li><span class="n">9</span><a href="#changelog">Changelog</a></li>
  </ol>
</nav>

<hr class="rule">

<section id="status">
  <h2><span class="sec-n">1</span>Status</h2>
  <div class="prose">
    <ul>
      <li><strong>Shipped (week-1 dependency):</strong> <code>vmf-measure</code> v0.1.0 — null quantiles, admissibility calculator, p ≥ 8192 log-Bessel, refusal mode. API and CSV schema frozen in <code>projects/kappa-measure/SCHEMA.md</code>; owner A. Plesner. Projects 1 and 5 can build against it now.</li>
      <li><strong>Validated:</strong> the D1 closed-form rel-RMSE surface against Monte Carlo on a 60-cell grid — every cell within a factor of 2 (section 4). The surface replaces the rectangular rule, which falsely rejects the best-conditioned cells in the program.</li>
      <li><strong>Audit scoped and pre-registered</strong> (commit <code>{PRE_REGISTRATION_COMMIT}</code>): only comparisons at unequal n and low κ/p. The artifact is real and large (section 5). One numerical correction to the brief, recorded there.</li>
      <li><strong>Not done:</strong> D3 nulls for Ethayarajh anisotropy / IsoScore / effective rank / participation ratio (the PR and effective-rank closed forms are textbook MP — budgeted ~5 core-hours); D5 upstream SciPy report; the 12–20-claim reproduction subset on original checkpoints.</li>
      <li><strong>Blocked:</strong> nothing.</li>
    </ul>
  </div>
</section>

<section id="tool">
  <h2><span class="sec-n">2</span>The tool</h2>
  <div class="prose">
    <pre class="code-block"><code>from vmf_measure import kappa_hat, rel_rmse_surface, null_quantile, min_resolvable_n

est = kappa_hat(r_bar=0.05, p=512, n=70)
est.status   # "not resolvable at this n"
est.kappa    # None — no number ships below the boundary
est.rel_rmse # 2.87 — predicted, so callers see how far below they are

min_resolvable_n(0.05, 512)          # ~1.3e4 samples needed for 1% rel-RMSE
null_quantile("kappa_hat", 512, 70, 0.95)  # matched uniform null</code></pre>
    <p>
      Refusal is the default and the refusal string is part of the frozen schema. Passing
      <code>n=None</code> opts out of the check explicitly; <code>refuse=False</code> returns the
      number with the predicted error attached. The rectangular rule
      (<code>R̄ &gt; 0.19 AND n/p ≥ 10</code>) is retired everywhere in the program: it denies
      p=512, R̄=0.85, n/p=0.5, where the measured relative RMSE is
      <strong>{fmt_pct(calib["empirical_rel_rmse"])}</strong> — one of the best-conditioned cells
      we have.
    </p>
  </div>
</section>

<section id="bessel">
  <h2><span class="sec-n">3</span>Log-Bessel at p ≥ 8192</h2>
  <div class="prose">
    <p>
      SciPy's <code>special.ive</code> underflows to exactly zero above order ≈511, which makes
      even the <em>admissibility formula</em> uncomputable in stock SciPy at p = 4096
      (<code>ive(2047, κ)</code> is NaN for κ ≲ 2048). <code>vmf_measure.log_iv</code> is a NumPy
      translation of the CUSF piecewise algorithm already tested in this repo
      (<code>src/torch_log_iv.py</code>), checked here against a 50-digit mpmath reference:
    </p>
    <div class="table-scroll"><table class="data-table">
      <thead><tr><th>order ν (p = 2ν+2)</th><th>κ</th><th>rel. error vs mpmath</th><th>SciPy <code>ive</code></th></tr></thead>
      <tbody>
        <tr><td class="rowhead">511 (p=1024)</td><td class="num">512</td><td class="num">≤1e-15</td><td class="cell-warn">underflow begins</td></tr>
        <tr><td class="rowhead">2047 (p=4096)</td><td class="num">2048</td><td class="num">2.1e-16</td><td class="cell-crit">NaN</td></tr>
        <tr><td class="rowhead">4095 (p=8192)</td><td class="num">100 – 10000</td><td class="num">≤2.1e-16</td><td class="cell-crit">0 / NaN</td></tr>
        <tr><td class="rowhead">8191 (p=16384)</td><td class="num">8500</td><td class="num">1.9e-16</td><td class="cell-crit">0 / NaN</td></tr>
        <tr><td class="rowhead">16383 (p=32768)</td><td class="num">20000</td><td class="num">≤1e-15</td><td class="cell-crit">0 / NaN</td></tr>
      </tbody>
    </table></div>
  </div>
</section>

<section id="surface">
  <h2><span class="sec-n">4</span>D1: closed-form surface vs Monte Carlo</h2>
  <div class="prose">
    <div class="equation">rel-RMSE(κ̂) ≈ hypot( amp·(1−R²)/(2nR²),  1/(κ√(n·A_p′(κ))) )
amp = 1 + 2R²/(1−R²) − 2R²/(p−R²)</div>
    <p>
      Each cell draws up to 2000 independent vMF datasets (Householder NumPy sampler,
      PCG64DXSM) and compares the empirical relative RMSE of the joint-MLE κ̂ against the
      surface evaluated at the population R̄ = A_p(κ). Range of emp/pred across p ∈
      {{128, 512, 2048}} per (κ/p, n/p) band; the pre-registered acceptance target was a
      factor of 2.
    </p>
    <div class="table-scroll"><table class="data-table">
      <thead><tr><th>κ/p</th><th>n/p</th><th>emp/pred range over p</th></tr></thead>
      <tbody>
{surf_rows}
      </tbody>
    </table></div>
    <div class="callout">
      <strong>Calibration cell confirmed.</strong> p=512, κ/p=3.0 (R̄=0.85), n/p=0.5:
      empirical <strong>{fmt_pct(calib["empirical_rel_rmse"])}</strong> vs predicted
      {fmt_pct(calib["predicted_rel_rmse"])} — the brief's quoted 0.63%. The rectangular rule
      would refuse this cell at less than half the samples it demands.
    </div>
    <div class="callout warn">
      <strong>Where the surface misses, it misses safe.</strong> The sub-2× cells are all at
      κ/p ≤ 0.2, deep inside the refusal region: the κ̂ distribution saturates, so the surface
      <em>over</em>estimates the error (emp/pred down to {ratio.min():.2f}). The refusal
      decision is unchanged there — both numbers are far above any target — but dependent
      projects should not read predicted rel-RMSE &gt; 1 as a quantitative error estimate.
    </div>
  </div>
</section>

<section id="audit">
  <h2><span class="sec-n">5</span>Audit: comparisons at unequal n and low κ/p</h2>
  <div class="prose">
    <p>
      The census framing is not run — the yield model in the brief gives 0–2 consequential
      findings, for three structural reasons: a common upward bias makes ordinal comparisons
      at matched n <em>conservative</em>; infeasibility lives at κ/p ≤ 0.2, where an upward
      bias makes "near-isotropic" conclusions conservative too; and the flagship target
      already ships a matched null. The surviving attack surface is a comparison at
      <strong>unequal n</strong> in the low-κ/p regime — rare vs frequent tokens, small vs
      large classes. Verdicts are pre-registered (SCHEMA.md, commit
      <code>{PRE_REGISTRATION_COMMIT}</code>) and phrased “not resolvable as reported at the
      original n”, never “wrong”.
    </p>
    <div class="table-scroll"><table class="data-table">
      <thead><tr><th>claim</th><th>κ/p</th><th>n_A / n_B</th><th>true ratio</th><th>observed κ̂ ratio</th><th>verdict</th></tr></thead>
      <tbody>
{audit_rows}
      </tbody>
    </table></div>
    <p>{plot}</p>
    <div class="callout crit">
      <strong>Correction to the brief.</strong> The quoted spurious ratio of 3.01 at κ/p = 0.05,
      n_A=70 vs n_B=700 does not reproduce: the measured value is
      <strong>{canonical["ratio_observed"]:.2f}</strong>, stable across p ∈ {{64, 512, 2048}} and
      across estimators (MLE and Banerjee agree to 3 s.f. in this regime). The artifact's
      magnitude is bounded by √(n_B/n_A) = √10 ≈ 3.16, approached only at the uniform null;
      3.01 is reached near κ/p ≈ 0.017, not 0.05. The structural conclusion — this is the one
      surviving attack surface — stands.
    </div>
    <div class="callout">
      <strong>The matched-n controls survive, as predicted.</strong> Identical true κ at equal
      n gives ratio {control["ratio_observed"]:.3f}; a true ratio of 2.0 in a resolvable cell
      reads {true2["ratio_observed"]:.3f}; and the brief's own illustration — true ratio 2.0,
      both arms at n=100, low κ/p — reads <strong>{illustration["ratio_observed"]:.3f}</strong>,
      exactly the predicted 1.267: the bias shrinks gaps, it does not inflate them. Note the
      matched-n low-κ/p control is <em>individually</em> refused (each arm's absolute κ̂ is
      meaningless) while its ratio is exactly 1 — refusal guards absolute claims; ordinal
      claims at matched n were never at risk.
    </div>
  </div>
</section>

<section id="nulls">
  <h2><span class="sec-n">6</span>Nulls</h2>
  <div class="prose">
    <p>
      Three matched nulls, definitions frozen in SCHEMA.md: <strong>uniform</strong>
      (closed-form: R̄² ~ χ²_p/(np), exact E[R̄²]=1/n), <strong>mean_removed</strong>
      (sample-mean direction projected out, renormalized), and
      <strong>covariance_matched</strong> (Gaussian with the data's empirical covariance,
      normalized to the sphere). Full grid in <code>results/null_quantiles.csv</code>
      ({len(nulls)} rows).
    </p>
    <div class="callout">
      <strong>The mean-removed null is near-degenerate — a finding, not a bug.</strong>
      Projecting out the sample-mean direction makes the un-normalized resultant vanish
      <em>exactly</em> (Σy_i = 0 by construction), so the renormalized R̄ null is second-order:
      q95 = {mr:.4f} against {kop_uniform:.4f} for the uniform null at p=512, n=70 — a
      {kop_uniform / mr:.0f}× gap. Any pipeline that mean-centers before normalizing lives on
      this null, and judging it against √(1/n) would manufacture concentration out of thin air.
    </div>
    <p>
      Pending (D3): Ethayarajh anisotropy, IsoScore, effective rank, participation ratio on the
      same grid. For isotropic data, PR = p/(1 + p/n) reproduces simulation to three decimals —
      textbook Marchenko–Pastur, budgeted ~5 core-hours to confirm and cite, not a contribution.
    </p>
  </div>
</section>

<section id="card">
  <h2><span class="sec-n">7</span>Geometry measurement card (D4)</h2>
  <div class="prose">
    <p>Every geometry claim in the program ships with this card. The canonical audit cell, filled in:</p>
    <div class="table-scroll"><table class="data-table">
      <tbody>
        <tr><th>instrument</th><td>κ̂, joint vMF MLE (vmf-measure {surface["tool_version"].iloc[0]})</td></tr>
        <tr><th>n, p</th><td class="num">n = 70, p = 512</td></tr>
        <tr><th>null model</th><td>uniform on S⁵¹¹ (matched); q95(κ̂) available in null_quantiles.csv</td></tr>
        <tr><th>observed</th><td class="num">R̄ ≈ {canonical["r_bar_a"]:.4f}, κ̂ ≈ {canonical["kappa_hat_a"]:.1f} (true κ = {canonical["kappa_true_a"]:.1f})</td></tr>
        <tr><th>predicted rel-RMSE</th><td class="num">{canonical["rel_rmse_a"]:.2f} (target 0.01)</td></tr>
        <tr><th>admissible</th><td class="cell-crit">no — “not resolvable at this n”</td></tr>
        <tr><th>verdict</th><td>not resolvable as reported at the original n</td></tr>
      </tbody>
    </table></div>
  </div>
</section>

<section id="scope">
  <h2><span class="sec-n">8</span>Scope &amp; caveats</h2>
  <div class="prose">
    <ul>
      <li><strong>CPU-only, float64, single machine</strong> (Apple M3 Max) for every number on this page. Monte Carlo cells use one seed each with 60–2000 replicates; precision comes from replication against a deterministic formula, not from seed sweeps.</li>
      <li><strong>The census was not run, deliberately.</strong> The yield model retains 0–2 consequential findings from 60–120 harvested claims; section 5 covers the one attack surface that survives the three structural filters.</li>
      <li><strong>Refusal ≠ ordinal safety.</strong> vmf-measure refuses per-arm, on absolute error. An ordinal comparison at matched n can be valid where both arms refuse; the tool does not adjudicate ordinal claims.</li>
      <li><strong>Audit verdicts are simulation-based.</strong> The 12–20-claim reproduction subset on original checkpoints is pending; substituting checkpoints would make it a different experiment.</li>
      <li><strong>D5 is pending.</strong> The SciPy <code>ive</code> underflow (exact 0 above order ≈511; NaN for <code>ive(2047, κ)</code>, κ ≲ 2048) still needs its upstream report and the log-Bessel contribution.</li>
      <li><strong>Numbers may move.</strong> Dependents pin the schema, not the values; null quantiles will tighten as MC reps grow.</li>
    </ul>
  </div>
</section>

<section id="changelog">
  <h2><span class="sec-n">9</span>Changelog</h2>
  <div class="prose">
    <ul>
      <li><strong>{today}</strong> — v0.1.0 shipped: frozen API/schema (<code>SCHEMA.md</code>), log-Bessel validated to p=32768 against mpmath-50dps, surface validated on a 60-cell MC grid, audit run on pre-registered verdicts (commit <code>{PRE_REGISTRATION_COMMIT}</code>). One correction to the brief recorded (§5). Report HEAD <code>{head_hash}</code>.</li>
    </ul>
  </div>
</section>

<footer class="report-foot">
  <p><strong>Reproduce:</strong> <code>uv run python projects/kappa-measure/scripts/validate_surface.py</code>, <code>audit_unequal_n.py</code>, <code>make_null_quantiles.py</code>, then <code>make_report.py</code>. Tests: <code>cd projects/kappa-measure/tests &amp;&amp; uv run python -m unittest discover -s . -v</code>.</p>
  <p><strong>Raw data:</strong> <code>projects/kappa-measure/results/*.csv</code> (frozen schemas). API contract: <code>projects/kappa-measure/SCHEMA.md</code>.</p>
</footer>
</div>
</body>
</html>
"""

    out = PROJECT_ROOT / "report.html"
    out.write_text(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
