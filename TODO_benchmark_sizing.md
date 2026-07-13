# TODO: increase the benchmark sampling size

**Status: not done — reminder.**

`benchmark.py` currently times every backend on a **fixed `NUM_SAMPLES = 5000`**
per `sample()` call (`TARGET_TIME_S = 5.0` only controls how many timeit loops
are run, not the size of one call). At `n = 5000` the per-call **fixed
overhead** — QR rotation setup, array allocation, Python/dispatch cost — is a
large fraction of the runtime at low/mid dimension, so the reported throughput
is **well below the steady-state rate**.

## Evidence

Comparing this repo's scipy CSVs (`measurements/benchmark_scipy_dim*.csv`,
float64, kappa=50, n=5000) against the *same* scipy sampler measured at large,
calibrated sizes (the AgenticOptimisation verifier — one call ≈ several
seconds, so overhead is amortised):

| dim | here (n=5000) | large-n (steady state) | under-report |
|----:|--------------:|-----------------------:|:------------:|
| 2   | 4.28 M/s      | 13.25 M/s              | 3.1×         |
| 8   | 2.79 M/s      | 5.51 M/s               | 2.0×         |
| 16  | 1.78 M/s      | 3.37 M/s               | 1.9×         |
| 32  | 0.98 M/s      | 1.69 M/s               | 1.7×         |
| 64  | 0.385 M/s     | 0.629 M/s              | 1.6×         |
| 128 | 0.154 M/s     | 0.158 M/s              | 1.0×         |
| 256 | 0.051 M/s     | 0.033 M/s              | ~1× (differs by machine/BLAS) |

The gap shrinks as dim grows (per-sample work dominates), but at dim ≤ 64 the
n=5000 numbers understate throughput by 1.6–3×. The torch/numpy backends have
their own fixed overheads too, so cross-backend comparisons at n=5000 are also
skewed.

## What to do

Make the per-call sample count large enough to reach steady state. Options,
easiest first:

1. **Raise `NUM_SAMPLES`** to something large (e.g. 1–5 M) — crude but fixes the
   low-dim bias immediately. Watch memory at high dim (output = n·dim·8 bytes).
2. **Calibrate `NUM_SAMPLES` per (dim, kappa)** so one call takes ~5 s of a
   *reference* backend, capped by a memory budget (output ≲ 1.2 GB/call), and
   repeat the call to fill the timeit window when the cap binds. This is what
   the AgenticOptimisation verifier does (`benchmark/verifier/verify.py`,
   `bench()`), and is the recommended approach for apples-to-apples numbers.

Either way, re-run and regenerate `measurements/*.csv` and
`benchmark_analysis.ipynb`.

## Also: special-case dim=2 and dim=3 (much faster)

The samplers here (`vmf_torch_hh`, `vmf_numpy_hh`, `vmf_torch`, `vmf_numpy`, …)
run the general Wood (1994) rejection sampler + a north-pole→mu rotation for
**every** dimension. scipy does **not** — it special-cases the two low dims,
which is much faster because it avoids both the rejection loop and the rotation:

- **dim = 2** — the vMF distribution reduces to the **von Mises distribution on
  the circle**. scipy draws the angle directly with `random_state.vonmises(
  mean_angle, kappa, size)` where `mean_angle = arctan2(mu[1], mu[0])`, then
  returns `(cos θ, sin θ)`. No rejection, no rotation. (scipy `_rvs_2d`.)
- **dim = 3** — the polar coordinate `x = cos(angle to mu)` has a closed-form
  inverse CDF, so no rejection is needed:
  `x = 1 + ln(u + (1 - u) · e^(−2κ)) / κ`, with `u ~ U(0,1)`; then
  `(y, z) = sqrt(1 − x²) · uniform_on_circle`, and rotate to `mu`.
  (scipy `_rvs_3d`; ref: Wenzel Jakob's vMF note.)

Recommendation: add these two closed-form paths to the samplers here (dispatch
on `dim == 2` / `dim == 3` before the general rejection code). At dim 2–3 the
general path is markedly slower than scipy; the closed forms remove that gap.

> Note: the AgenticOptimisation benchmark copy (`benchmark/examples/sol_torch_hh.py`)
> has been fixed to include these special cases. This repo's source was left
> unchanged on purpose — this note is the reminder to apply the same fix here.

_(Filed while wiring the AgenticOptimisation vMF `rvs` benchmark, which reuses
this repo's `vmf_torch_hh` as a strong baseline.)_

## From human: we do not really need the second householder flip
We do two householder flips which has the benefit that the space keeps the same orientation, but this is not strictly needed and can be dropped. There are no actual benefits from this.
We also need to update the description in the artifact. And it should be a standalone html file rather than a claude artifact as I would want to use the results for my blog.
