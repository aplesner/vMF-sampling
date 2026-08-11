# Project 6 — Measuring representation geometry: nulls, feasibility, and a working estimator

**Read `00-shared-constraints.md` first.**

## The question

In the (n, p) regime modern representations actually occupy, when is a geometry measurement
**resolvable at all** — and what does a practitioner do about it?

This is an instrument-and-calibration project with an audit appendix. It is **not** a census of the
literature (see "Scope honesty" below).

## Why it exists

Three established facts from this repo's own benchmarking:
- SciPy's Bessel path is unusable at p ≥ 1024: `ive` underflows to exactly zero above order ~511, so
  Sra's estimator returns NaN and safeguarded Newton **silently stalls at ~1e-3**. `ive(2047, κ)`
  returns NaN for κ ≲ 2048 — **the admissibility formula itself is uncomputable in stock SciPy at
  p = 4096.** That last fact is the strongest argument for shipping a tool, and it is better evidence
  than any audit finding.
- Statistical error dominates numerical error by **~12 orders of magnitude** at high p.
- κ/p governs accuracy, and the finite-sample bias is systematically **upward** — the same direction
  as any "more concentrated than uniform" claim.

## Deliverables

**D1 — the closed-form feasibility surface.** Replace the rectangular rule (`R̄ > 0.19 AND n/p ≥ 10`)
with:

```
rel-RMSE(κ̂) ≈ hypot( amp·(1−R²)/(2nR²),  1/(κ√(n·A_p′(κ))) )
amp = 1 + 2R²/(1−R²) − 2R²/(p−R²)
```

Matches Monte Carlo to 2 s.f. across the grid. The rectangle is grossly conservative in the
high-concentration corner — at p=512, R̄=0.85, n/p=0.5 the true relative RMSE is **0.63%**, better
than the 1% target at half the samples the rule demands — so a rectangle-based audit produces false
positives, and the audit output *is* the classification.

**D2 — `vmf-measure`.** Null quantiles (uniform, mean-removed, covariance-matched), the admissibility
calculator, a log-Bessel path valid at p ≥ 8192, and a **refusal mode as default**: `kappa_hat`
returns "not resolvable at this n" below the boundary rather than a number. Ship an API and table
schema **frozen by end of week 2 with a named owner** — Projects 1 and 5 need it in week 1 and will
otherwise build throwaways. Numerical values may move after the freeze; dependents pin the schema.

**D3 — nulls for the other instruments.** Average cosine, Ethayarajh anisotropy, IsoScore, effective
rank, participation ratio, over the same (n, p) grid. This is what makes the project about *geometry
measurement* rather than about vMF.

**D4 — a reporting standard.** A one-page "geometry measurement card" (state n, p, null, CI,
admissibility) that reviewers can demand.

**D5 — upstream fix.** Report the SciPy `ive` underflow and contribute the log-Bessel path.

## Scope honesty — what this project must not claim

**Do not run the 60–120-claim census with an "N published claims are wrong" headline.** The yield
model, worked through with measured numbers:

| stage | retained |
| --- | --- |
| harvested claims | 60–120 |
| report (n, p) precisely enough to classify | 35–80 |
| sit at κ/p ≤ 0.2, the only infeasible region | 6–20 |
| absolute, not ordinal-at-matched-n | 2–8 |
| bias points *toward* the claim | **1–4** |
| consequential — flipping it changes a headline | **0–2** |

Three structural reasons the flagship framing collapses:

1. **The bias shrinks gaps rather than inflating them.** A common upward bias makes ordinal claims
   conservative: at n=100, a true κ ratio of 2.0 reads as 1.267. Ordinal comparisons at matched n
   survive, and even different-p comparisons at matched n survive (rel. bias 1.66% at p=512 vs 1.71%
   at p=2048).
2. **Infeasibility lives entirely at κ/p ≤ 0.2** — which is exactly where a paper's conclusion is
   "near-isotropic / low concentration," so an upward bias there makes the author's own conclusion
   *conservative*. Where the bias is claim-aligned ("representations are anisotropic," "the class
   collapsed"), κ/p ≥ 0.5 and the bias is under 3% even at n/p = 0.13. **The proposition "the bias
   direction matches the claim it supports" is true only in the regime where the bias is negligible.**
3. **The flagship target already ships a matched null.** Ethayarajh (2019) §3.4 defines
   `Baseline(f_ℓ)` and subtracts it, and its headline claims are ordinal. The narrow real crack: that
   baseline is estimated from only 1K samples, so it carries its own sampling error — one sentence,
   not a headline.

**What to run instead:** the surviving attack surface is **comparisons at unequal n and low κ/p** —
rare-token vs frequent-token, small-class vs large-class, low-resource vs high-resource. At κ/p=0.05,
n_A=70 vs n_B=700 produces a spurious κ̂ ratio of **3.01 at identical true κ**. That is real, it is
publishable, and it is about a tenth the size of the claimed flagship. Scope the audit to it.

Report every verdict as **"not resolvable as reported at the original n,"** never as "wrong."
Pre-register inclusion criteria and verdict definitions before harvesting, with a commit hash.
Mandatory fields per claim: instrument, ordinal-vs-absolute, n_A vs n_B, aggregation over classes.
Keep a 12–20 claim reproduction subset **on the original checkpoints** — substituting checkpoints
makes it a different experiment.

## Retired — do not reintroduce

**The "known-mean estimator gives a √p extension, 64× at p=4096" claim.** Both estimators have null
fluctuation of order √(p/n); `p/√n` is the *mean* of the estimated-mean null, a deterministic offset
a matched null subtracts exactly, so √p is a ratio of one estimator's bias to the other's standard
deviation. Measured detection thresholds at α=0.05, power 0.5: known-μ crossover ~1.65×√(p/n) vs
Rayleigh ~11–13×, i.e. **≈7.4× at p=4096, not 64×** — overstated by a factor of 8.6. For
*estimation* accuracy the gain is **exactly zero**: κ and μ are orthogonal parameters, `I_κκ =
n·A_p′(κ)` in both cases. And with a sample-split μ it is **worse than plain Rayleigh** (power 0.075
vs 0.110 at p=4096, n=100, κ=4×√(p/n)).

The one place an a-priori μ genuinely exists is the **classifier weight vector w_c evaluated on
held-out features** (NC3 self-duality) — that belongs to Project 5, not here.

**The Marchenko–Pastur analysis is correct but is not a contribution.** For isotropic data,
participation ratio and effective rank are both biased **downward** — toward "low-dimensional,
anisotropic" — and `PR = p/(1 + p/n)` reproduces simulation to three decimals (0.333/0.500/0.666/
0.909 at n/p = 0.5/1/2/10). This is textbook MP, it is a one-line deterministic correction, and
shuffle controls are already standard in the neural-dimensionality literature. **Budget ~5 core-hours
to confirm the closed forms and cite MP** — not the 100–150 core-hours proposed. Spend the difference
on the κ̂ surface, where the correction is *not* closed-form.

## Applications worth running

- **Attention-head diagnostic.** Per-head κ̂ across released models, against prefix-matching-score
  labels. Note the ground truth is itself a heuristic, so a failure could indict the taxonomy rather
  than κ̂ — state which interpretation you will take **before** running. Watch the `n = number of
  heads` trap: do not fit κ̂ over a sample whose size is the head count.
- **Pythia training trajectory.** Layerwise κ̂(t) over ~20 of the ~150 checkpoints, crossed with the
  known emergence point of induction heads. Does directional concentration *precede* the rise in
  prefix-matching score? Gate the ~20 GPU-h of forward passes on the free embedding trajectory
  showing structure first.
- **Cross-generation embedding atlas**, with one pre-registered falsifiable hypothesis tested only in
  resolvable cells — otherwise it is a motivating figure, not a finding.

## Budget

~40 GPU-h + ~150 CPU-core-h. Cheapest project in the program and a dependency for two others, so it
runs first.
