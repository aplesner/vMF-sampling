# Experiment roadmap: vMF sampling and kappa estimation applied to S-VAE and nGPT

Source papers are local: `papers/vmf_vae.pdf` (Davidson et al. 2018, S-VAE),
`papers/ngpt.pdf` (Loshchilov et al. 2024, nGPT), `papers/sra-paper.pdf`.

## Why these two papers connect to this repository

The S-VAE's KL term depends only on `kappa`, and its analytic gradient
(their Eq. 6) is built from the Bessel ratio

```text
A_p(kappa) = I_(p/2)(kappa) / I_(p/2-1)(kappa)
```

which is exactly the quantity `src/vmf_kappa_torch.py` already computes for
concentration estimation. The S-VAE's sampler is Ulrich (1984) with a
Householder reflection mapping `e_1` to `mu` -- the same algorithm as
`src/vmf_torch_hh.py`. Their only statement on cost is an acceptance-rate
table (1.0--1.46 proposals per sample, *improving* with dimension); they report
no wall-clock, batching, or GPU numbers.

nGPT normalizes every embedding and weight matrix to the unit sphere but never
models a distribution on it. Its geometry diagnostics are three ad-hoc proxies:
vector norms, weight-matrix condition numbers, and pairwise dot-product
histograms. Fitting a vMF and reporting `kappa_hat` is a better-specified
version of the same analysis, and is cheaper: `O(V d)` for the mean resultant
length versus `O(V^2)` for a pairwise dot-product histogram at `V = 32000`.

## Two capability gaps to close first

Both are prerequisites for most of the list.

**Gap A -- per-row `(mu_i, kappa_i)` batched sampling.** `TorchvMFHH` takes a
scalar `kappa` and a single `mu`. Every VAE and per-token-noise experiment needs
`(B, d)` means with `(B,)` concentrations in one call. The envelope constants
`a, b, d` become `(B,)` tensors, the rejection loop needs mask-and-refill rather
than compaction, and the Householder update becomes row-wise. Estimated 1--2
days.

**Gap B -- numerically stable differentiable vMF KL at large `m`.** The S-VAE
KL and its kappa-gradient need `I_(m/2)/I_(m/2-1)`, `I_(m/2+1)/I_(m/2-1)`,
`I_(m/2-2)/I_(m/2-1)`, and `log I_(m/2-1)`, all expressible through `log_iv`
differences. The original S-VAE code uses SciPy's `ive` and cannot reach
`m` in the thousands (see the measured breakdown below). A
`torch.autograd.Function` with an analytic backward is roughly 1 day, and it is
what makes an `m = 1024` S-VAE possible at all.

## Measured constraints from `scripts/benchmark_kappa.py`

Two results from this repository's own benchmark bound what the experiments
below can claim.

**SciPy's Bessel path fails for `p >= 1024`.** `scipy.special.ive` underflows to
exactly zero for order `>= ~511`, so `log I_v` becomes `-inf`. Sra's two-step
and the SciPy Brent reference return `NaN`; safeguarded Newton falls back to
bisection and stalls near `1e-3` relative error. Compiled CUSF `log_iv` stays at
`~1e-13` across the whole grid. Any experiment at `d_model >= 1024` must use the
CUSF or pure-torch log-Bessel path.

**Statistical reliability, not solver precision, is the binding constraint.**
Median relative error of `kappa_hat` against the generating kappa at
`n = 10000` samples, identical across every accurate solver:

| `p` | `kappa=1` | `kappa=10` | `kappa=50` | `kappa=1000` |
| --- | --- | --- | --- | --- |
| 64 | 0.17 | 4.8e-3 | 2.4e-3 | 9.2e-4 |
| 1024 | 9.3 | 0.43 | 2.0e-2 | 5.6e-4 |
| 4096 | 39.8 | 3.2 | 0.29 | 1.0e-3 |

At `p = 4096` and `kappa = 1`, `kappa_hat` is wrong by a factor of 40 with ten
thousand samples. Every reported `kappa_hat` on real embeddings therefore needs
a **matched null**: draw `n` uniform unit vectors in `R^p`, run the same
estimator, and report the observed value against the null distribution, which is
not centered at zero. Building that null over an `(n, p)` grid with enough
replicates for percentile intervals is where the sampler's throughput advantage
becomes a precondition rather than a convenience.

**The bias is systematically upward, and `kappa/p` governs it.** The reliability
study sweeps `n` from 100 to 100000. Smallest tested `n` reaching a given median
relative error:

| `p` | `kappa` | `kappa/p` | `n` for 10% | `n` for 1% |
| --- | --- | --- | --- | --- |
| 256 | 200 | 0.78 | 100 | 1000 |
| 256 | 10 | 0.04 | 10000 | 100000 |
| 1024 | 1000 | 0.98 | 100 | 1000 |
| 1024 | 50 | 0.05 | 10000 | 100000 |
| 1024 | 10 | 0.01 | 100000 | > 100000 |
| 4096 | 1000 | 0.24 | 100 | 1000 |
| 4096 | 200 | 0.05 | 10000 | 100000 |
| 4096 | 50 | 0.01 | > 100000 | > 100000 |

Once `kappa/p` is above roughly 0.2, about `n = 10-25 p` samples buy 1% accuracy.
Below `kappa/p ~ 0.05`, even `n = 100000` does not. The signed bias is positive in
essentially every cell, and it grows precisely as concentration falls -- so the
direction of the artifact is the same as the direction of the effect anyone
analyzing embeddings wants to claim ("more concentrated than uniform"). This
makes the matched null non-negotiable rather than good practice, and it puts a
hard feasibility bound on group A:

- GPT-2 / BERT scale, `p = 768` with `V = 50257`: `n/p ~ 65`, so `kappa` must be
  above roughly 75 for 1% accuracy.
- LLaMA scale, `p = 4096` with `V = 32000`: `n/p ~ 8`, so `kappa` must be above
  roughly 800. If embedding concentration is actually near 50, the estimate
  carries about 30% error.

Check the feasibility bound before running any group A experiment, and prefer
`E_output`/`E_input` (`n = V`) over per-token contextual fits (`n` = occurrence
count), which are far more exposed. A2's per-token variant and A3 need
frequency-stratified nulls for exactly this reason.

## Ordering

Cheap and unblocking first, then inference-only science, then training runs.

1. E1, E2 -- no dependencies, produce the null tables everything else needs.
2. Gap A and Gap B -- 2--3 days, unblock groups B, C, D.
3. A1--A4 -- inference-only, laptop scale, first real scientific output.
4. C1, C3 -- S-VAE validation harness and the sampler/estimator round trip.
5. E4 then C6 -- settle vMF versus Power Spherical with the variance mechanism
   understood first.
6. C2 -- extend the S-VAE past its reported dimension wall.
7. B1--B3 -- inference-only robustness; B2 is the cleanest demonstration that
   the sampler and the estimator compose.
8. E3 -- bias-corrected high-`p` estimator.
9. D1's precursor -- one day, potentially a striking cross-paper result.
10. B4, D2 -- highest ambition and highest risk.

---

## Group E -- methodology, no model required

**E1. Extend the S-VAE acceptance-rate table.** Their Table 5 is a 5x9 grid from
1000 Monte Carlo samples, stopping at `d = 100` and `kappa = 1e4`. Redo at `1e6+`
samples, extend to `d` up to 8192 and `kappa` up to `1e6`, and add the number
they omit: observed wall-clock per *accepted* sample. A 1.4x rejection overhead
is irrelevant next to the `O(nd)` Householder cost at `d = 4096`. Also verify
the `b`-underflow series fallback in `src/vmf_torch_hh.py` against the exact
envelope, since that guard activates precisely in the high-kappa regime and the
paper never discusses it. ~1 day.

**E2. The `(n, p)` bias surface for `kappa_hat`.** Map bias, MSE, and the null
distribution of `kappa_hat` over `p` in 2--4096, `n` in 10--1e6, true `kappa` in
0--1e4, with enough replicates for percentile intervals. Ship it as a
precomputed lookup table. Every experiment in groups A and B needs this null,
and `measurements/kappa/` currently only has `kappa=50` reference data. The
literature on kappa estimation at `p` in the thousands with realistic `n` is
thin, so the table is plausibly a contribution in itself. ~2--3 days, CPU
cluster.

**E3. Bias-corrected `kappa_hat` for high dimension.** From E2's surface, fit and
validate a correction in the spirit of the classical nearly-unbiased estimators
but for `p >> 1`. Deliverable: an estimator taking the sample size as an
argument, which none of the five current methods do. Highest ratio of novelty to
compute cost in the list. ~1 week.

**E4. Gradient-variance study of the S-VAE reparameterization.** The `g_cor`
correction term is a score-function-style estimator and the suspected variance
culprit at large `m` -- and the stated reason the Power Spherical distribution
exists. Measure the variance of the kappa-gradient from `g_rep + g_cor` versus
`g_rep` alone versus Power Spherical's pathwise gradient, over `m` in 2--4096 and
`kappa` in 1--1e4, using a synthetic objective. No model needed. Run this
*before* C6. ~2 days.

## Group A -- inference-only kappa analysis of released checkpoints

**A1. vMF atlas of embedding matrices.** Fit vMF to rows of `E_input` and
`E_output` across GPT-2, Pythia, LLaMA-2/3, Qwen. Report `R_bar`, `kappa_hat`
from all three fast estimators, and excess over the matched null. Plot against
`d_model` and parameter count. ~1 day.

**A2. Layerwise `kappa_hat` of hidden states.** Capture `h` at every layer over a
fixed corpus slice, normalize, fit per layer, pooled and grouped by token id.
Tests nGPT's central metaphor -- each layer contributes a displacement on the
sphere with `alpha ~ 0.2-0.3` -- which predicts a smooth monotone `kappa_hat`
trajectory in a normalized model versus a jumpier one in a standard GPT. The
paper never measures this. ~1--2 days.

**A3. Per-token `kappa_hat` as a lexical statistic.** Per vocabulary item,
collect contextual embeddings, fit vMF, correlate `kappa_hat` against unigram
frequency, WordNet sense count, part of speech, whole-word versus fragment, and
next-token entropy. Hypothesis: an unsupervised polysemy detector. Requires
frequency-stratified nulls -- a token seen 20 times and one seen 2e6 times have
very different bias. ~1 day.

**A4. `kappa_hat` as an attention-head diagnostic.** Recompute nGPT's Fig. 5
condition-number analysis with `kappa_hat` of the row-normalized per-head `W_q`,
`W_k`, `W_v`, and correlate against condition number and stable rank. Either it
tracks conditioning -- then it is cheaper than an SVD and comparable across
dimensions -- or it measures something orthogonal, which is more interesting.
~1 day.

**A5. Isotropy literature reconciliation.** The embedding-anisotropy literature
uses average cosine similarity, IsoScore, and participation ratios. A
null-calibrated `kappa_hat` is a principled replacement. Reproduce 3--4
published findings and check which survive the finite-sample bias correction.
Some published anisotropy may be `R_bar` sampling noise. ~2--3 days.

**A6. Mixture-of-vMF fit and per-cluster `kappa_hat`.** nGPT asserts embeddings
"form clusters". Test it: fit a `k`-component vMF mixture (Banerjee et al. 2005
EM), sweep `k`, model-select on held-out likelihood using the exact
`log C_m(kappa)`. The M-step *is* kappa estimation, once per component per
iteration, and the E-step needs `log C_m` at `m` in the thousands -- the CUSF
path. This is where fast batched estimation becomes load-bearing rather than
decorative. ~2--3 days.

**A7. `kappa_hat` along a training trajectory.** Pythia and OLMo release ~150
intermediate checkpoints. Plot `kappa_hat` of embeddings and hidden states versus
step and tokens. Does it follow a power law, saturate, and does saturation
coincide with the loss knee? nGPT's `s_z` mean grows with tokens (sharper
softmax); an analogous `kappa_hat` growth is the geometry-side counterpart.
~2 days.

**A8. Cross-model comparability note.** `kappa_hat` is dimension-aware where raw
cosine statistics are not: the same `R_bar` implies very different `kappa` at
`p = 768` versus `p = 4096`. Show concretely that ranking models by mean pairwise
cosine gives a different and wrong ordering. Small and clean. ~1 day.

## Group B -- robustness and vMF noise injection

**B1. vMF noise injection as an inference-time robustness knob.** At layer `l`,
replace `h` with a draw from `vMF(h/||h||, kappa)` and continue the forward pass.
Sweep `kappa` from 1e5 down to 10; measure perplexity, downstream accuracy, and
attack success rate against additive Gaussian noise at matched expected angular
displacement. On a normalized residual stream, Gaussian noise is the wrong
geometry -- it changes the norm, and post-`Norm` its angular magnitude is
dimension-dependent. vMF gives one interpretable dial, and `E[mu^T z] =
A_p(kappa)` converts `kappa` to a mean angular displacement in closed form using
existing code. Needs Gap A. ~2--3 days.

**B2. Calibrate the noise dial from the data.** Rather than sweeping blindly,
estimate `kappa_hat_l` per layer from A2 and inject at `c * kappa_hat_l` for
`c` in {0.5, 1, 2, 10}, so the perturbation is scale-matched to each layer's own
geometry. Report the robustness/accuracy Pareto frontier against uncalibrated
injection. This is the tightest single demonstration that the estimator and the
sampler compose. Needs Gap A and A2. ~1 day on top of B1.

**B3. Is `kappa_hat` an adversarial-input detector?** Compare per-layer
`kappa_hat` for adversarial, jailbreak, and prompt-injection inputs against
clean batches, plus the within-sequence version across token positions. Report
AUROC against max-softmax, Mahalanobis, and energy baselines. `kappa_hat` is one
scalar per layer per sequence, cheap enough for online use, unlike a Mahalanobis
distance needing a `d x d` covariance at `d = 4096`. ~2 days.

**B4. Randomized smoothing on the sphere.** Cohen et al.-style certified
robustness with vMF smoothing instead of Gaussian, yielding a geodesic rather
than `l_2` certificate. The Neyman-Pearson argument should go through since vMF
is monotone in `mu^T z`. Randomized smoothing needs 1e2--1e4 samples *per input*,
so at `d = 4096` the sampler's throughput is the difference between a real
experiment and a toy. Budget time for the certificate derivation in the
multi-class case. Needs Gap A. ~3--5 days.

**B5. Adversarial training with vMF augmentation.** Fine-tune a small LM or ViT
with vMF noise at a normalized layer, kappa annealed, against Gaussian-noise
augmentation and PGD at matched compute. First item needing a training run.
~1 week.

**B6. nGPT versus GPT under matched angular perturbation.** nGPT claims better
length extrapolation and well-conditioned matrices; whether it is more robust to
representation-space perturbation is untested. Train a small nGPT and a matched
GPT with `github.com/NVIDIA/ngpt` at nanoGPT scale, then apply the B1 protocol to
both. Self-contained question: does spherical parameterization confer angular
robustness, or only optimization speed? ~1--2 weeks.

## Group C -- recreating and extending the hyperspherical VAE

**C1. Faithful S-VAE reproduction.** Dynamically binarized MNIST, MLP
`[256,128]`/`[128,256]`, Adam 1e-3, batch 64, 100-epoch linear KL warm-up, early
stop look-ahead 50, log-likelihood by 500-sample importance weighting,
`d` in {2,5,10,20,40}, 10 runs. Targets: `d=2` LL -132.50+/-0.73; `d=40` LL
-90.87+/-0.34 (S) versus -88.93+/-0.30 (N). The baseline for all of group C and
the validation harness for Gaps A and B. Minutes per run. ~1--2 days.

**C2. Push past the reported dimension wall.** The paper stops at `d = 40` and
attributes degradation above `m ~ 20` to vanishing sphere surface area. With Gaps
A and B, extend to `d` in {64,...,1024} and measure log-likelihood,
reconstruction, KL, the distribution of learned `kappa(x)`, and the observed
acceptance rate. Three predictions to separate: (i) their Table 5 says acceptance
*improves* with `d`, so the wall is not a sampling-cost problem; (ii) their KL
grows 7.3 -> 33.5 over `d = 2 -> 40`, so check whether it simply swamps
reconstruction; (iii) check whether learned `kappa` saturates against a numerical
ceiling. Note the honest scope of (iii): SciPy's underflow begins near order 511,
i.e. `p ~ 1024`, so it cannot explain the originally reported `d = 20-40`
degradation -- it explains why nobody has pushed past ~40, not the wall itself.
~3--5 days.

**C3. Round-trip validation.** Train the S-VAE, sample `n` latents per `x` from
`q(z|x) = vMF(mu(x), kappa(x))` with the fast sampler, re-estimate with Sra's two
steps, and compare against the encoder's `kappa(x)`. Validates sampler,
estimator, and reparameterization gradient at once, and yields a reusable
calibration curve for how many samples each `d` needs. Cheap, exercises the whole
stack. ~1 day.

**C4. Does the aggregate posterior match the uniform prior?** The S-VAE's pitch
is that `U(S^(m-1))` is truly uninformative. Test the consequence: pool `z` over
the training set, fit a vMF to the aggregate posterior, and check whether
`kappa_hat_agg` is distinguishable from the matched uniform null. Sweep `d`; fit a
mixture to detect prior holes. The paper shows only Hammer-projection scatter
plots. ~1--2 days.

**C5. Fixed versus learned kappa, quantified.** The paper's criticism of a fixed
global kappa (Guu et al. 2018) is theoretical -- constant KL is never optimized.
Run the missing ablation: learned per-datapoint `kappa(x)` versus a swept fixed
global kappa, at every `d` up to 1024, measuring log-likelihood and the entropy
of the learned `kappa(x)` distribution. If learned kappa collapses to nearly
constant at large `d`, the bottleneck is "the extra parameter stops paying" rather
than "the sphere runs out of room". ~2 days.

**C6. vMF versus Power Spherical at matched compute.** Power Spherical
(arXiv:2006.04437) exists specifically because vMF rejection sampling was "slow
and numerically unstable in high dimensions". With the batched Householder
sampler and CUSF `log_iv`, re-run that comparison: same S-VAE, same `d` grid, swap
only the posterior family, report log-likelihood *and* wall-clock per epoch *and*
gradient variance. The most directly publishable item here -- it is a specific
claim in the literature that this repository's engineering may falsify. Either
vMF now wins, or Power Spherical wins on statistics alone, which isolates the
cause to `g_cor`'s gradient variance rather than to speed. Run E4 first.
~3 days.

**C7. Product-of-spheres S-VAE.** Implement the arXiv:1910.02912 decomposition
`S^M0 x ... x S^Mk` with one kappa per factor, on static MNIST, Omniglot, and
Caltech 101 Silhouettes. Reproduce their static-MNIST result (`S^40` LL -96.32 ->
4-way product -92.65) and extend to total ambient dimension 256--1024 with many
factors. `k` factors means `k` simultaneous samplers with different
`(mu_j, kappa_j)` and dimensions -- exactly the Gap A workload. The follow-up is
a workshop note with limited scale; scaling it is low-risk. ~3--5 days.

**C8. S-VGAE link prediction.** Reproduce their Table 4 (Cora 92.7 -> 94.1 AUC,
Citeseer 90.3 -> 94.7, Pubmed 97.1 versus 96.0). Then close their loose end: they
*omitted* `d_z = 64` for S-VGAE because it ran out of memory, and Pubmed, the
largest graph, is the one dataset where S-VGAE loses. Run `d_z` in {64,128,256}
with the allocation-conscious in-place sampler and determine whether the Pubmed
loss is a capacity limit or the hyperspherical bottleneck. Add OGB-arxiv for
scale. An explicit "we could not run this" that this implementation removes.
~2 days.

## Group D -- nGPT-flavored training experiments

**D1. vMF as an explicit output distribution for nGPT.** nGPT computes logits as
`z = E_output h * s_z` with learned `mean(s_z) ~ 60.8`, and its Table 4 shows
`s_z` is by far the most sensitivity-critical scaling factor (+3.12% validation
loss if mis-set). But `E_output h` is a vector of cosine similarities between unit
vectors, so `softmax(s_z * E_output h)` *is* a mixture-of-vMF likelihood with a
shared concentration: `s_z` is a concentration parameter that the paper does not
recognize as one. Make it explicit, initialize it from the geometry via the
estimator, and check whether the sensitivity disappears. ~1--2 weeks on 8 GPUs at
nanoGPT scale.

*Cheap precursor, do this first (inference only, ~1 day).* On a released nGPT
checkpoint, test numerically whether `mean(s_z) ~ 60.8` matches `kappa_hat` fit
to the output-embedding geometry at `d_model = 1024`. This repository's own
simulation confirms the test is well-conditioned: at `p = 1024`, `kappa = 60.8`
implies population `R_bar = 0.0592`, and a 32000-row vocabulary recovers
`kappa_hat = 61.06 +/- 0.15`, a 0.43% relative error. A null result would
therefore be informative rather than noise.

**D2. Does nGPT's residual stream follow a vMF random walk?** nGPT's mechanism is
`h <- Norm(h + alpha(h_A - h))` with `alpha ~ 0.2-0.3`. Model the layer
transition as `h_(l+1) ~ vMF(h_l, kappa_l)`, estimate `kappa_l` from empirical
transitions, and test the analytic prediction that `kappa_l` is determined by
`alpha_(A,l)` and `alpha_(M,l)` alone. If the fit holds, the result is a
generative surrogate of the entire forward pass as a vMF random walk on
`S^(d-1)` with known per-layer concentration -- which can then be *sampled* to
test how much of nGPT's behavior the walk explains. The most intellectually
interesting item here, and precisely the explicit spherical distribution nGPT
lacks. Moderate risk the fit is poor; still informative. ~3 days if checkpoints
are released.

**D3. vMF-regularized embedding training.** Penalize deviation of the embedding
matrix's `kappa_hat` from a target -- a differentiable "hold embeddings at
concentration `kappa*`" regularizer, requiring the Bessel-ratio gradient from Gap
B. Compare against Wang and Isola's uniformity loss, which is a Gaussian-kernel
energy rather than a directional likelihood. `kappa* -> 0` recovers the
uniformity regime; the open question is whether a nonzero target, matching
nGPT's own observation that embeddings *should* cluster, beats pure uniformity.
~1 week.

**D4. Eigen learning rates versus measured geometry.** Train small nGPTs at
several depths and context lengths and test whether learned `alpha_A, alpha_M`
are predictable from the per-layer `kappa_hat` trajectory of A2. The paper
reports `alpha_A` 0.25 -> 0.20 as depth goes 24 -> 36, and 0.245 -> 0.258 as
context goes 1k -> 8k. If a simple relation holds, the eigen learning rates can
be initialized or fixed from a cheap geometric measurement -- and their Table 5
already shows single scalars cost under 0.3%. ~1 week.

**D5. Replicate the nGPT speedup at small scale.** Prerequisite infrastructure
for D1 and D4. Use `github.com/NVIDIA/ngpt` at nanoGPT scale on OpenWebText at 1k
context. Be explicit about the caveat: the 4x/10x/20x figures are at 0.5B/1B on
64 A100s, and the reported 80%/60% per-step overhead means wall-clock gain is
materially below token gain. Reproducing the ordering is realistic; reproducing
the factors is not. ~1 week.
