# Project 4 pre-registration — Gate G1 (restated) and success criteria

Project: `projects/hyperspherical-vae/` — hyperspherical VAEs in hyperspherical
architectures (brief: `docs/projects/04-hyperspherical-vae.md`).

## Why v1 Gate G1 had to be replaced

v1 G1 required, across the dimension sweep: **mean learned κ falling
monotonically with d while KL still rises.** KL(vMF(μ,κ)‖U) at fixed κ is
non-monotone in m, peaking at m* ≈ 0.762 κ (verified numerically, V2 below).
The conjunction is therefore satisfiable only while the posterior stays on the
rising branch, κ/m ≳ 1/0.762 ≈ 1.31 — equivalently ρ = A_m(κ) ≳ 0.69.
Extrapolating Davidson's implied κ(d) trend (V3) crosses that boundary between
d = 64 and d = 128, so at d ≥ 256 the v1 gate fails for reasons unrelated to
whether the science is right. Inverting Davidson's KL column is only possible
with the restated quantities at all past m ≈ 1024, because SciPy's `ive`
underflows to exactly zero there (V5).

## Restated Gate G1 (this supersedes the v1 text)

Notation: m = d + 1 (brief's degrees-of-freedom convention); per-datapoint
ρ(x) = A_m(κ(x)) ∈ [0,1); KL/m = per-coordinate rate in nats.

- **G1a (scale-free de-concentration).** ρ̄(d) = mean_x A_m(κ(x)) is
  non-increasing in d across the sweep, up to Monte Carlo standard error.
  A_m is strictly increasing in κ at every m, so ρ̄ falling is exactly "κ
  falling faster than m grows" — the scale-free content of v1's "κ falls",
  with no branch dependence.
- **G1b (rate still paid).** KL/m at the learned posterior stays ≥ r₀ = 0.02
  nats/dim at every d. This rules out the degenerate reading of the dimension
  wall as posterior collapse to the uniform prior (KL = 0). The floor is far
  below Davidson's d = 40 operating point (KL/m = 0.82) and below every point
  on his sweep (min 0.82); KL = m·f(ρ) with f strictly increasing (V1), so
  G1b is a one-dimensional check on ρ̄: f(ρ̄) ≥ r₀ ⇔ ρ̄ ≥ ρ(r₀) ≈ 0.2.
- **G1c (branch report — measured event, not pass/fail).** Record the sign of
  ρ̄(d) − ρ* with ρ* = 0.69 at every d. Crossing below ρ* is *expected* near
  m ≈ 90–120 on the Davidson trend and is precisely the phenomenon that made
  the v1 gate unsatisfiable; it must appear in the report, not in the gate.

**G1 passes iff G1a ∧ G1b.** The conjunction is jointly satisfiable on both
branches of KL(m): G1a asks ρ̄ to fall, G1b asks it to stay above ≈0.2, and
nothing else. The v1 gate's second conjunct ("raw KL rises") is exactly the
statement "the model stays on the rising branch", which is a property of the
(m, κ) trajectory and not of the science; it is replaced by G1b's absolute
floor and reported separately via G1c.

## Pre-registered numerical facts behind the restatement

Verified by `scripts/verify_g1.py` (float64, CUSF/torch log-Bessel path;
14 unit tests in `tests/test_vmf_kl.py`):

- **V1 — KL = m·f(ρ) at fixed ρ.** KL/m is constant in m to relative spread
  ≤ 1e-3 for ρ ≤ 0.3 across m = 8…4096 (≤ 2.5e-2 at ρ = 0.5, converging as m
  grows). Anchor: ρ = 0.3 gives KL/m = 0.04715 at m = 128, 1024, 4096
  (brief's claimed value reproduced).
- **V2 — branch boundary.** KL(m) at fixed κ peaks at m*/κ = 0.762 ± 0.001
  for κ ≥ 20; at the peak ρ* = A_{m*}(κ) = 0.689 ± 0.002. So "KL rises at
  fixed κ" ⟺ ρ ≳ 0.69, independent of κ — the v1 gate is a gate on ρ in
  disguise.
- **V3 — Davidson inversion.** Inverting Table 1's S-VAE KL column at
  m = d + 1 gives implied κ/m = 657, 150, 52.5, 15.5, 4.63 at
  d = 2, 5, 10, 20, 40 (ρ = 0.9995 … 0.900). His entire sweep sits deep on the
  rising branch. Trend fit log κ = 8.476 − 0.880 log m crosses the branch
  boundary κ/m = 1.31 between d = 64 and d = 128. (These correct the brief's
  "4.6e5, 527, 82, 18.0, 4.85" column, which appears to use a different m
  convention at small d; the qualitative claim — rising branch throughout — is
  unchanged.)
- **V4 — v1 gate is mathematically dead at d ≥ 128.** On the extrapolated
  trend, KL(m, κ̂(m)) decreases from d = 128 onward while κ̂ still falls: the
  v1 conjunction fails no matter what the encoder learns.
- **V5 — computability.** At the trend operating point, scipy's `ive(order, κ̂)`
  is exactly 0 for m ≥ 1024 (order ≥ 511); the torch/CUSF log-Bessel path
  keeps A_m, KL, and both inversions finite through m = 8192. KL evaluation
  includes a large-κ asymptotic branch (switch κ* = 8.2e3·m^{3/4}, first-order
  correction, worst jump 2.6e-3 nats) covering κ/m up to ~1e5 at small m.

## Success criteria for the tiers (pre-registered before runs)

Pre-registration commit: `c55a906`.

- **T1 (validity, gates T2/T3).** Reference SciPy-based S-VAE fails to produce
  a finite loss at m ∈ {1024, 4096}; this repo's path trains with finite loss
  everywhere. Davidson MNIST reproduction at d ≤ 40: S-VAE and N-VAE LL
  (IW-500) within 2 s.d. of Table 1.
- **T2.** MDE: with 5 seeds × per-arm LR sweeps, detect an S-VAE-vs-N-VAE LL
  gap of 1 nat at d ≥ 256 against the pooled within-seed s.d. (to be measured
  in the pilot); if the pilot s.d. makes 1 nat undetectable, the estimand
  switches to the ordinal claim (rank order across arms at matched d).
- **T3.** Primary contrast: nViT-VAE (vMF) minus arm-4 control (conventional
  ViT + vMF + √d rescale), CIFAR-10, d ∈ {64, 256, 1024}, 3 seeds. If arm 4
  closes the gap (difference within 2 s.e. of zero), the result is reported as
  an input-scaling story.
- **T4.** Matched-rate diagonal Gaussian is the primary comparison, not an
  ablation.
- **T5.** Collapse diagnostic is the effective rank/dispersion of the
  aggregate {μ(x)} against a matched null, not a κ histogram.

Commit hash of this pre-registration: **c55a906** (the criteria precede every
run; the hash necessarily follows the commit that contains this file).

## Two-GPU protocol (T2/T3)

Two RTX 3090s (tikgpu[06,07,09], `--gres=gpu:geforce_rtx_3090:1`) as
**independent workers**: one (arm × d × seed) run per card, never DDP. LR
sweeps are per arm; no arm is ever compared at a shared LR. Runs are assigned
to cards by queue order; assignment is recorded per run so CRN pairing uses
seed, not card.
