"""Regenerate projects/hyperspherical-vae/report.html from results/*.csv.

Follows the repo convention (scripts/make_blog.py): the report is generated,
self-contained, and inherits the design tokens of docs/kappa_report.html.
Run after scripts/verify_g1.py.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vmf_kl import kl_vmf_uniform  # noqa: E402

STYLE = """
  :root {
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
    --border: rgba(11, 11, 11, 0.10); --accent: #2a78d6;
    --s1: #2a78d6; --s2: #1baf7a; --s3: #4a3aa7;
    --code-bg: #f2f1ed; --callout-bg: #f2f5fa;
    --ok-bg: rgba(27, 175, 122, 0.13);   --ok-ink: #0f6b49;
    --warn-bg: rgba(214, 158, 42, 0.16); --warn-ink: #7d5a06;
    --bad-bg: rgba(214, 108, 42, 0.17);  --bad-ink: #8a3d09;
    --crit-bg: rgba(201, 52, 52, 0.16);  --crit-ink: #92211f;
    --note-bg: #f4f2ee;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
      --border: rgba(255,255,255,0.10); --accent: #3987e5;
      --s1: #3987e5; --s2: #199e70; --s3: #9085e9;
      --code-bg: #232322; --callout-bg: #1d2430;
      --ok-bg: rgba(25, 158, 112, 0.22);  --ok-ink: #5fd6a8;
      --warn-bg: rgba(214, 158, 42, 0.20); --warn-ink: #e0b356;
      --bad-bg: rgba(214, 108, 42, 0.22);  --bad-ink: #ef9d63;
      --crit-bg: rgba(201, 52, 52, 0.24);  --crit-ink: #f0847f;
      --note-bg: #1e1e1d;
    }
  }
  body { background: var(--page); font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         color: var(--ink); line-height: 1.6; font-size: 15.5px; margin: 0; }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 0 24px 96px; }
  .prose { max-width: 780px; }
  header.report-head { padding: 56px 0 8px; }
  .eyebrow { font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase;
             color: var(--muted); margin: 0 0 14px; font-weight: 600; }
  h1 { font-size: 30px; line-height: 1.2; margin: 0 0 10px; }
  h2 { font-size: 21px; margin: 40px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
  h3 { font-size: 16px; margin: 24px 0 8px; }
  .lede { color: var(--ink-2); font-size: 17px; margin: 0; }
  table { border-collapse: collapse; margin: 14px 0; font-size: 13.5px; width: 100%; }
  th, td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--border);
           font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--ink-2); font-weight: 600; border-bottom: 1px solid var(--baseline); }
  code { background: var(--code-bg); padding: 1px 5px; border-radius: 4px; font-size: 13px; }
  .status { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 10px; margin: 22px 0; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
          padding: 12px 14px; font-size: 13.5px; }
  .card .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
             color: var(--muted); font-weight: 600; margin-bottom: 4px; }
  .pill { display: inline-block; border-radius: 999px; padding: 1px 10px; font-size: 12px;
          font-weight: 600; }
  .pill.ok { background: var(--ok-bg); color: var(--ok-ink); }
  .pill.warn { background: var(--warn-bg); color: var(--warn-ink); }
  .pill.bad { background: var(--bad-bg); color: var(--bad-ink); }
  .callout { background: var(--callout-bg); border-radius: 8px; padding: 14px 18px;
             margin: 16px 0; font-size: 14.5px; }
  .figure { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
            padding: 16px; margin: 16px 0; }
  .figure svg text { fill: var(--muted); }
  .figcap { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
  .changelog { font-size: 13.5px; color: var(--ink-2); }
  .changelog li { margin-bottom: 4px; }
"""


def read_csv(name: str) -> list[dict]:
    with (PROJECT_ROOT / "results" / name).open() as fh:
        return list(csv.DictReader(fh))


def fmt(x: float, digits: int = 4) -> str:
    if abs(x) >= 1e4 or (x != 0 and abs(x) < 1e-3):
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".") if "." in f"{x:.{digits}f}" else f"{x:.{digits}f}"


def branch_figure() -> str:
    """KL(m) at fixed kappa, peak marked; x-axis rescaled to m/kappa."""
    W, H, L, R, T, B = 560, 300, 56, 14, 18, 42
    def pt(x, y):
        return (L + x / 3.0 * (W - L - R), H - B - y / 1.05 * (H - T - B))
    parts = []
    for kappa, color in [(20, "#2a78d6"), (100, "#1baf7a"), (400, "#4a3aa7")]:
        ms = torch.linspace(2.0, 3.0 * kappa, 600, dtype=torch.float64)
        kl = kl_vmf_uniform(kappa, ms)
        kmax = kl.max().item()
        coords = [pt(m / kappa, k / kmax) for m, k in zip(ms.tolist(), kl.tolist())]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        px, py = pt(float(ms[int(torch.argmax(kl))].item()) / kappa, 1.0)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>')
    bx, _ = pt(0.762, 0)
    parts.append(f'<line x1="{bx:.1f}" y1="{T}" x2="{bx:.1f}" y2="{H-B}" stroke="#898781" stroke-dasharray="4 3"/>')
    parts.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#898781"/>')
    parts.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#898781"/>')
    for xv, lbl in [(0, "0"), (0.762, "0.76"), (1, "1"), (2, "2"), (3, "3")]:
        x, _ = pt(xv, 0)
        parts.append(f'<text x="{x:.1f}" y="{H-B+16}" font-size="11" text-anchor="middle">{lbl}</text>')
    for yv in [0, 0.5, 1.0]:
        _, y = pt(0, yv)
        parts.append(f'<text x="{L-6}" y="{y+4:.1f}" font-size="11" text-anchor="end">{yv}</text>')
    parts.append(f'<text x="{W/2}" y="{H-6}" font-size="12" text-anchor="middle">m / &#954;</text>')
    parts.append(f'<text x="{L+4}" y="{T+8}" font-size="12">KL(m, &#954;) / max</text>')
    legend = " ".join(f'<tspan fill="{c}">&#954;={k}</tspan>' for k, c in [(20, "#2a78d6"), (100, "#1baf7a"), (400, "#4a3aa7")])
    parts.append(f'<text x="{W-R-4}" y="{T+10}" font-size="12" text-anchor="end">{legend}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" '
            f'aria-label="KL versus m at fixed kappa">{"".join(parts)}</svg>')


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    inv = read_csv("g1_davidson_inversion.csv")
    branch = read_csv("g1_branch.csv")
    old_gate = read_csv("g1_old_gate_failure.csv")

    inv_table = table(
        ["d", "m=d+1", "KL (Davidson T.1)", "&#954;&#772; (inverted)", "&#954;/m", "&#961; = A_m(&#954;&#772;)", "KL/m"],
        [[r["d"], r["m"], r["kl_reported"], f'{float(r["kappa_implied"]):.1f}',
          f'{float(r["kappa_over_m"]):.2f}', f'{float(r["rho"]):.5f}',
          f'{float(r["kl_over_m"]):.3f}'] for r in inv],
    )
    branch_table = table(
        ["&#954;", "m* (argmax KL)", "m*/&#954;", "&#961;* = A_m*(&#954;)", "KL/m at peak"],
        [[f'{float(r["kappa"]):g}', f'{float(r["m_star"]):.1f}',
          f'{float(r["m_star_over_kappa"]):.4f}', f'{float(r["rho_star"]):.4f}',
          f'{float(r["kl_over_m_at_peak"]):.4f}'] for r in branch],
    )
    gate_table = table(
        ["d", "&#954;/m (trend)", "&#961; (trend)", "KL on trend", "v1 gate: KL rising?"],
        [[r["d"], f'{float(r["kappa_over_m"]):.3f}', f'{float(r["rho"]):.4f}',
          f'{float(r["kl_trend"]):.2f}',
          ('<span class="pill ok">yes</span>' if r["on_rising_branch"] == "True"
           else '<span class="pill bad">no — conjunct dead</span>')]
         for r in old_gate],
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project 4 — Hyperspherical VAEs in hyperspherical architectures</title>
<style>{STYLE}</style>
</head>
<body>
<div class="wrap">
<header class="report-head">
  <p class="eyebrow">Project 4 &middot; hyperspherical VAEs</p>
  <h1>Hyperspherical VAEs in hyperspherical architectures</h1>
  <p class="lede">Is Davidson et al.&rsquo;s dimension wall a property of the spherical
  latent, or of the mismatch between a spherical latent and an unnormalized
  encoder/decoder? This page tracks the gate decisions first, then tier results.</p>
</header>

<h2>Status</h2>
<div class="status">
  <div class="card"><div class="k">Gate G1 (restated)</div>
    <span class="pill ok">fixed &amp; verified</span>
    <div>Restated in terms of &#961; = A_m(&#954;) and KL/m; numerically verified,
    14 unit tests green. See below.</div></div>
  <div class="card"><div class="k">T1a — numerical validity</div>
    <span class="pill ok">confirmed</span>
    <div>Reference SciPy S-VAE numerics are non-finite at encoder-realistic
    &kappa; from m = 512 up (9/10 rows at m &ge; 1024); ours finite everywhere,
    3.7e-11 agreement where SciPy works.</div></div>
  <div class="card"><div class="k">T1b — MNIST reproduction</div>
    <span class="pill warn">harness ready</span>
    <div>d &le; 40, S-VAE/N-VAE, runs on the T2 array; needs cluster (unreachable
    at last attempt).</div></div>
  <div class="card"><div class="k">T2 — dimension sweep</div>
    <span class="pill warn">not started</span>
    <div>d &isin; &#123;5&hellip;1024&#125; &times; 4 arms &times; 5 seeds, per-arm LR sweeps.</div></div>
  <div class="card"><div class="k">T3 — nViT-VAE (core new experiment)</div>
    <span class="pill warn">not started</span>
    <div>vMF latent in nViT encoder/decoder vs 4 controls incl. the &#8730;d-rescale
    control that decides the story. CIFAR-10, d &isin; &#123;64, 256, 1024&#125;, 3 seeds,
    two 3090s as independent workers.</div></div>
  <div class="card"><div class="k">T4/T5 — product-of-spheres, collapse</div>
    <span class="pill warn">not started</span>
    <div>Matched-rate diagonal Gaussian as primary; collapse diagnosed via
    aggregate-&#956; effective rank vs matched null.</div></div>
</div>

<h2>Gate G1 — restated and verified</h2>
<div class="prose">
<p><strong>v1 text (unsatisfiable):</strong> &ldquo;mean learned &#954; falls monotonically
with d <em>while KL still rises</em>.&rdquo; KL at fixed &#954; is non-monotone in m, peaking
at m* &asymp; 0.762&#954;; the conjunction is only satisfiable on the rising branch,
&#954;/m &gtrsim; 1.31. On the extrapolated Davidson trend that boundary is crossed
between d = 64 and d = 128, so at d &ge; 256 the v1 gate fails regardless of the
science.</p>
<p><strong>Restated gate</strong> (full text in <code>PREREGISTRATION.md</code>):
<strong>G1a</strong> &#961;&#772;(d) = mean_x A_m(&#954;(x)) non-increasing in d;
<strong>G1b</strong> KL/m &ge; r&#8320; = 0.02 nats/dim at every d (rules out the wall being
mere collapse to the uniform prior; f(&#961;) = KL/m is strictly increasing, so G1b
&#8660; &#961;&#772; &gtrsim; 0.2); <strong>G1c</strong> (report, not pass/fail) sign of
&#961;&#772; &minus; &#961;* per d, with &#961;* = 0.69. G1 passes iff G1a &and; G1b.
The conjunction is jointly satisfiable on both branches of KL(m).</p>
</div>

<h3>V2 — the rising branch is a statement about &#961;</h3>
<div class="figure">{branch_figure()}
<div class="figcap">KL(vMF(&#954;) &#8214; Uniform) as a function of m at fixed &#954;
(curves normalized to their max). The peak sits at m*/&#954; = 0.762 across two
decades of &#954; (dashed line), i.e. at &#961;* = A_m*(&#954;) &asymp; 0.69. Raw KL
rises with m iff the posterior keeps &#961; above &#961;* — which is exactly what a
de-concentrating encoder stops doing.</div></div>
{branch_table}

<h3>V3 — Davidson&rsquo;s sweep, inverted (m = d+1)</h3>
<div class="prose"><p>vMF KL depends on &#954; alone, so his Table 1 KL column inverts
exactly. Implied &#954;/m = 657 &rarr; 4.63 from d = 2 to d = 40, with &#961; &ge; 0.90
throughout: his entire sweep is deep on the rising branch, and his &#961;&#772; falls
monotonically. Trend fit log &#954; = 8.476 &minus; 0.880 log m. These values correct the
brief&rsquo;s &ldquo;4.6e5, 527, 82, 18.0, 4.85&rdquo; column (a different m convention at
small d); the qualitative claim is unchanged.</p></div>
{inv_table}

<h3>V4 — the v1 gate dies at d &ge; 128 on the trend</h3>
{gate_table}

<h3>V1 &amp; V5 — the restated quantities are sound and computable</h3>
<div class="prose">
<p>KL = m&middot;f(&#961;) at fixed &#961;: relative spread of KL/m across
m = 8&hellip;4096 is &le; 1e-3 for &#961; &le; 0.3 (anchor: &#961;=0.3 gives 0.04715 at
m = 128, 1024, 4096 — the brief&rsquo;s value), &le; 2.5e-2 at &#961; = 0.5 and converging
in m. At the trend operating point, scipy&rsquo;s <code>ive(order, &#954;&#770;)</code> is exactly
0 from m = 1024 upward, while the CUSF/torch log-Bessel path keeps every gate
quantity finite through m = 8192. KL evaluation switches to a first-order
asymptotic form at &#954;* = 8.2e3&middot;m<sup>3/4</sup> (worst jump at the switch 2.6e-3
nats); round-trip inversion tests are exact. Full log:
<code>results/g1_verification_log.txt</code>; CSVs under <code>results/</code>.</p>
</div>

<h2>T1a — the reference S-VAE is numerically invalid at high m</h2>
<div class="prose"><p>Reproducing the exact numerics of Davidson&rsquo;s reference
implementation (KL via <code>scipy.special.ive</code>, eq.-6 ive-ratio gradient):
the forward KL and its gradient go non-finite at the &kappa; values an S-VAE
encoder actually emits (softplus init &asymp; 0.54, trend &kappa; &asymp; 3&ndash;20)
from m = 512 upward, and in 9/10 probed (&kappa;, m) cells at m &ge; 1024 — the
loss is NaN before training starts. Our CUSF/torch log-Bessel path agrees with
the reference to 3.7e-11 (max rel) wherever the reference is finite, and stays
finite through m = 4096 (tested through 8192 in the unit tests). Full grid:
<code>results/t1_numerical_validity.csv</code>.</p></div>

<h2>Two-GPU protocol (T2/T3)</h2>
<div class="prose"><p>Two RTX 3090s as independent workers — one
(arm &times; d &times; seed) run per card, never DDP. Every arm gets its own LR sweep;
no arm is compared at a shared LR. Card assignment is recorded per run so CRN
pairing uses seed, not card.</p></div>

<h2>Changelog</h2>
<ul class="changelog">
  <li><strong>2026-07-27</strong> — Project scaffolded. Gate G1 restated in terms of
  &#961; = A_m(&#954;) and KL/m (supersedes the unsatisfiable v1 conjunction); V1–V5
  verified numerically; 14 unit tests green. T1–T5 not started.</li>
  <li><strong>2026-07-27 (2)</strong> — Differentiable per-row batched vMF sampler
  (Ulrich + Naesseth/Davidson reparameterization; analytic &kappa; gradients for
  KL and log C_m) with 10 statistical tests green; T1a confirmed (reference
  SciPy path NaN at encoder-realistic &kappa; from m = 512); T1/T2 harness
  (4 arms, IW-500, gradient-variance decomposition, aggregate-&mu; diagnostics),
  Slurm array for two 3090s as independent workers. Cluster unreachable at
  last attempt — T1b/T2 submission pending.</li>
</ul>
</div>
</body>
</html>
"""
    (PROJECT_ROOT / "report.html").write_text(html)
    print(f"wrote {PROJECT_ROOT / 'report.html'}")


if __name__ == "__main__":
    main()
