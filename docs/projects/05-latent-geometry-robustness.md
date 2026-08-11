# Project 5 — Angular perturbation of classifier latents

**Read `00-shared-constraints.md` first.**

## The question

Calibrated vMF noise injected into a network's latents is a **perturbation dose with a closed-form
readout** (`E[μᵀz] = A_p(κ)`). Using it as an instrument: how does angular displacement propagate
layer to layer, and do adversarial examples displace latents *more* than their accuracy cost
predicts — or do they sit on the same dose–response curve as ordinary corruption?

One question. This is a measurement-instrument project, not a defense paper.

## What changed from v1

v1 bundled four things — directional measurement of latents, adversarial detection, normalized
architectures as a robustness intervention, and latent noise. That is four papers' worth of claims in
one section, and three of them are weak. Triage:

- **Latent noise as an instrument — promote to the core.** Nobody else can compute calibrated
  high-dimensional vMF noise cheaply; the readout is closed-form; it is falsifiable either way; and
  it depends on none of the contested feasibility claims.
- **Adversarial detection — one section, framed as a negative.** Tramèr et al. (ICML 2022) show a
  robust detector at ε implies a robust classifier at ε/2, so any interesting-operating-point
  detection claim implies a classifier beyond SOTA. Lead with that as the reason the paper does
  *not* make a security claim.
- **Normalized architectures as a robustness intervention — cut**, after one correctly-designed gate.
- **The vMF randomized-smoothing certificate — cut as a security claim**, keep one paragraph of math.

## Two claims from v1 that are numerically false — do not reintroduce

1. **"A converged CIFAR-100 class at n=500 < p=512 is unestimable."** Measured relative RMSE is
   **0.63%**. At high concentration the bias floor is ≈1/n and does not involve p, so "n < p" carries
   no information once R̄ is large. The intended negative result does not exist.
2. **The KappaFace audit target.** At p=512, n≈70, κ/p ~1–3: rel. RMSE 1.9–2.9%, Spearman(true, κ̂)
   = **0.995**. And KappaFace **z-scores κ̂ across classes** before using it, so any bias common to
   all classes cancels by construction. κ there is a training-loop control signal whose correctness
   criterion is IJB-C/LFW accuracy, not an estimand — attacking it with measurement theory is a
   category error that hands a reviewer the framing.

## The constructive reframe

The defensible measurement claim inverts v1's: **κ̂ is well-conditioned exactly where the trace-ratio
NC1 statistic is not.** The trace ratio needs `Σ_W Σ_B⁺`, whose pseudo-inverse is singular at
n_class < p — which is the standard CIFAR-100/Tiny-ImageNet regime. A likelihood-based directional
statistic is O(p) and well-posed at any n ≥ 2. That is a positive, constructive result rather than a
demolition, and it is the honest version of the NC1 contribution.

Keep the **p-sweep at fixed backbone and fixed n** — varying only the head width and holding
everything else constant is the direct empirical test that κ/p, not p, governs estimability, and it
is the most defensible new measurement in the project. Frame it as validation-on-trained-networks of
Project 6's theory; cross-cite.

## Experiments

**T1 — angular dose–response and amplification (~5 GPU-h, the core).** ResNet-18 (+ normalized
variant) on CIFAR-10, 3 seeds. At each layer ℓ with measured κ̂_ℓ, inject vMF noise at
κ_inject = c·κ̂_ℓ for c ∈ {0.25, 0.5, 1, 2, 4, 10}; measure (i) accuracy vs angular dose (closed-form
via A_p), (ii) downstream angular displacement at layers ℓ′ > ℓ, giving a **layer-to-layer angular
amplification matrix**. Then compute the per-layer angular displacement induced by PGD/AutoAttack
examples and by CIFAR-10-C at matched accuracy degradation, and express each as a **noise-equivalent
dose**.

The result is informative in both directions: if adversarial examples displace latents more than
their noise-equivalent accuracy cost predicts, that is a genuine directional signature; if they sit
on the same curve as corruption, that is Tramèr's picture in geometric form and should be reported
as such, with no detection claim drawn from it.

**Prerequisite:** per-row (μᵢ, κᵢ) batched sampling is not yet in the repo. Scope that first — the
estimate ranges from 1–2 days to 2–3 weeks and it gates everything here.

**T2 — the p-sweep (~4 GPU-h).** Fixed backbone, fixed n, head width p ∈ {64,128,256,512,1024,2048}.
Report κ̂ with matched nulls in every cell against the closed-form RMSE surface.

**T3 — detection, Tramèr-compliant (~9 GPU-h), reported as a negative.** ResNet-18 + ViT-S on
CIFAR-10/100, **encoders shared across all detectors** (v1 confounded detector with seed).
- Baselines: keep Mahalanobis and energy, **add LID and feature squeezing** — the adversarial-
  detection literature's own reference points, absent from v1, which used OOD-era baselines only.
- **Corruption control (mandatory):** CIFAR-10-C at matched clean-accuracy degradation. Without it
  the metric cannot distinguish "detects attack" from "detects distribution shift," and either
  answer is a finding.
- **Held-out attacks:** AutoAttack + Square, never used for tuning or layer-aggregation choice.
  Declare the split in the repo before running.
- **Adaptive attack:** likelihood-penalized PGD over 5 λ plus a detector-aware C&W variant, EOT-20
  wherever injection is active. **Report post-adaptive AUROC as the headline number.**

The defensible positive claim is narrow and worth stating: the directional score is O(p) and
well-posed at any n ≥ 2, exactly where class-conditional Mahalanobis has a singular covariance —
matching on AUROC while conferring no adaptive-attack security.

**T4 — the robustness gate (~11 GPU-h), expected to fail.** v1's Gate G0 gates on a robust-accuracy
difference between two **non**-adversarially-trained ResNet-18s, which at ε=8/255 is either 0.0% vs
0.0% or exactly the gradient-masking artifact another clause of the same gate exists to catch. Fix:
hoist a **Fast-AT pair** (FGSM-RS, 50 epochs, catastrophic-overfitting monitor) into the gate, give
**each arm its own 5-point LR sweep** — v1 never sweeps the AT arms, making the headline comparison
an untuned-baseline comparison in a program whose stated rule forbids them — then run the full
masking screen on those models. **State the MDE: ~2.5–4.7 points at 3 seeds, versus a literature-
typical claimed effect of 1–3 points.** Only the masking screen blocks; the rest is descriptive.

**Pre-registered kill condition:** gap below MDE at tuned LRs, or any masking diagnostic failing in
≥2/3 seeds → the intervention arm is cancelled permanently and ships as a falsification paragraph. A
clean, LR-controlled, masking-screened falsification of "hyperspherical normalization confers
adversarial robustness" is useful — the claim is repeated widely and tested rarely — but it is a
section, not a paper.

## On the smoothing certificate

The math correction is right and worth one paragraph: the vMF likelihood ratio is
`exp(κ(μ₁−μ₂)ᵀz)`, monotone in a linear functional, so Neyman–Pearson worst-case sets are half-spaces
intersected with the sphere and Cohen's two-point argument carries to a geodesic radius; and the
needed object is the CDF of `aᵀz` for a **general** direction a, a 2-D integral (radial `t = μᵀz`
times a Beta-distributed transverse coordinate), not the 1-D marginal of μᵀz that v1 assumed.

But drop the security claim. Three reasons: a **geodesic ball is not an L2 ball** — an L2 ball of
radius 0.85 around a CIFAR image spans ‖y‖ ∈ [28.15, 29.85], a 2.9% norm swing a spherical
certificate does not touch, so it certifies only norm-preserving perturbations, which is not a threat
model anyone uses. **It is numerically identical to Cohen (2019)** at σ_pix = ‖x‖/√κ — the claimed
0.502/0.669/0.836 radii are reproduced exactly by `3.196 × σ_pix`. And **non-vacuity holds only at
p_A = 1**: at p_A = 0.9 the radius is 0.20–0.33 L2, below the project's own 0.25 standing rule.

## Metrics

AUROC / AUPR / TPR@FPR for detection, **post-adaptive**. Angular displacement in radians with
closed-form dose. Robust accuracy under AutoAttack at RobustBench protocol with the reduced-N caveats
stated. Every κ̂ against a matched null.

## Budget

~30 GPU-h for T1–T3, +11 for the gate. Well inside envelope; the binding constraint is the per-row
sampler engineering.
