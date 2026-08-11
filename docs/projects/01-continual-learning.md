# Project 1 — Normalized networks in class-incremental learning

**Read `00-shared-constraints.md` first.**

## The question

Does architecture-wide hyperspherical normalization (nGPT-style: all weights and representations on
the unit sphere, learnable eigen-learning-rates, cosine classifier) improve the stability–plasticity
frontier in **rehearsal-free** class-incremental learning — and if so, is bounded angular motion of
representations the reason?

## Where this starts

A year-old informal result, to be replicated first: an nGPT-style normalized MLP on CIFAR-10 trained
on 5 classes, head extended by 5 rows, then trained on the held-out 5 with no rehearsal. The nMLP
converged **faster on the new classes and forgot less on the old ones**, and a 3D PCA showed nMLP
embeddings staying far more stable during phase 2 while the unnormalized MLP twisted the whole space.

The 5+5 CIFAR-10 split is non-standard and the MLP substrate has no spatial structure. Replicate it
to confirm the phenomenon, then move to standard protocols — do not build the paper on it.

## What changed from v1

- **The "rotation-clamped SGD" dial is cut.** Rescaling each layer's update so no fan-in vector
  rotates more than τ *is LARS* — a per-row adaptive step-size rule. Off the sphere, weight rotation
  and step size are the same quantity up to the radial component, so the intervention cannot separate
  "sphere" from "step size"; it is a step-size intervention wearing a geometric label. It also clamps
  the wrong object: the mediator is **representation** drift, and the map from fan-in weight rotation
  to feature displacement is depth-compounding, data-dependent, and non-injective.
- **The real mediation experiment, which nobody proposed:** a differentiable penalty on old-class
  **feature** displacement, applied at matched step size. This intervenes on the actual mediator and
  is the only design that can license "controlling variable, not correlate."
- **Do not cut the ResNet-50/ImageNet tier.** It is the only source of p=2048 features, hence the
  only empirical instance of the ν=1023 SciPy-underflow finding and the "wider is less measurable"
  result — which v1 designates as what survives if everything else fails.
- **The sequence-length premise was wrong.** "Drift compounds with T but head-magnitude imbalance
  does not" is contradicted by the literature — old/new class-weight norm ratio degrading with
  increment count is exactly what BiC (CVPR 2019) and Weight Aligning (CVPR 2020) document. Also,
  T ∈ {2,5,10,20,50} is not achievable on one dataset without confounding classes-per-task,
  data-per-task, and total gradient steps simultaneously.
- **Linear-mode-connectivity barriers are not comparable across parameterizations.** For the nMLP,
  weights live on unit-norm slices, so linear interpolation leaves the constraint manifold; the
  correct path is geodesic and the two are not on the same scale. Restrict any such comparison to
  *within-arm*.
- **Drift-after-task-2 predicting end-of-sequence forgetting is autocorrelation**, not prospective
  prediction — early drift is a partial sum of the total. The non-trivial version: within a fixed
  arm and config, across seeds only, does residual early drift beat *early forgetting itself* as a
  predictor?
- **Matched-new-class-accuracy conditioning is a collider.** New-class accuracy is a common effect of
  (arm, LR, latent per-run plasticity); forgetting also depends on that latent. Run comparisons on
  the frontier (frontier AUC, dominance fraction), not at a matched-accuracy point.

## Experiments

**Tier 0 — replicate and gate (~10 GPU-h).** nMLP vs tuned MLP on the 5+5 CIFAR-10 split, per-arm LR
and weight-decay sweeps, 5 partitions × 3 seeds. Gate: ≥3.0-point median paired gain with CI
excluding 0, **and** frontier dominance on ≥60% of the epoch grid.

**Tier 1 — the effective-learning-rate controls.** This is the part that decides whether the result
is real. All are cheap:
- Extend the existing angular-update matching (tune the baseline's λ so its per-layer
  ‖Δw‖/‖w‖ curves overlay the nMLP's) into a **5-point dose–response using weight decay** — the
  theoretically grounded ELR knob (van Laarhoven; Li–Arora). ~2 GPU-h, not 24.
- **LARS/LAMB as a scale-invariant-optimizer reference arm.** ~0.2 GPU-h. This is the cheapest
  remaining ELR control and it was silently dropped from v1.
- BN-stats-frozen and GroupNorm ResNet controls (removing BN otherwise hands nMLP a free win).
- Cosine-classifier-only arm and imprinting head-init sweep, to separate the head from the backbone.

**Tier 2 — the mediation experiment (the linchpin).** Differentiable penalty on old-class feature
displacement during phase 2, at matched step size, swept over 5 penalty strengths. If forgetting
moves monotonically with realized *feature* drift under an assigned penalty, the mediator claim is
established. Run at MLP scale first (~8 GPU-h); only escalate to ResNet-18 if it moves.

**Tier 3 — prevention vs compensation, the head-to-head nobody has run (~8 GPU-h).** SDC (Yu et al.,
CVPR 2020) accepts drift and corrects prototypes post hoc; normalization prevents it architecturally.
Grid: {baseline, nResNet} × {none, SDC}. Drop LDC — it trains a projector, which needs its own LR
sweep. **Caveats to honor:** SDC presupposes NCM/prototype evaluation, so all six arms must be
evaluated that way and it must be reported; and SDC needs **new-class features under the old model**,
which the current caching plan does not produce. Report gap-closure fraction normalized by the
available gap, with the estimator's own null. **Pre-registered kill condition:** if SDC recovers
~nothing on *either* arm, the mechanism story is wrong — do not narrate it as support.

**Tier 4 — standard protocols.** Split-CIFAR-100 10×10, Split-Tiny-ImageNet, then ImageNet-1k CIL
with ResNet-18 and ResNet-50 (FFCV). Keep the PTM tier (ImageNet-R + PTM-CIFAR-100) against
L2P/DualPrompt/CODA-Prompt/RanPAC/EASE at published schedules.

## Datasets

CIFAR-10 (replication only) → CIFAR-100 → Tiny-ImageNet → ImageNet-1k → ImageNet-R + PTM-CIFAR-100.
Add **ImageNet-A** for the PTM tier (~40 GPU-h) — pre-commit to spending it; 2026 reviewers ask.

## Metrics

Stability–plasticity **frontier** as the primary object; endpoint-only comparisons are banned. Report
both A_T and Ā with the conflation named; BWT and RWalk-forgetting with the max-vs-last distinction
stated; intransigence against a **matched-budget** joint oracle; per-order and per-seed variance
separated.

**Fix the plasticity metric:** "steps to 90% of its own converged accuracy" is self-normalized and
flatters an arm that converges to a lower plateau. Report time-to-fixed-absolute-threshold alongside.

Drift measured as null-calibrated angular displacement of old-class features between phases — against
a **training-noise null** (two independent runs' jitter), which is a different null from the uniform
one and is this project's own measurement design.

## Baselines

Tuned finetuning, EWC, LwF, LUCIR, PODNet, SSRE, FeTrIL, FeCAM, SimpleCIL, RanPAC. **FeTrIL and
FeCAM must run at every tier, not just MLP scale** — they are post-hoc feature-space methods,
essentially free on cached features, and having them appear only where they cannot threaten the claim
is indefensible. **Add SDC/LDC**: they are cited in the motivation and never run, and they are the
direct competitor to the drift story. At ImageNet include one rehearsal reference row (iCaRL or
DER++, ~13 GPU-h) — the field's tables always carry an upper reference.

## Budget

~400 A6000-h committed, ~10 h to open the first gate, ~20 h for all of Tier 1 + Tier 2's MLP half.
Verify FFCV throughput on one epoch before committing Tier 4.
