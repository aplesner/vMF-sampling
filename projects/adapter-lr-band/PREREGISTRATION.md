# Pre-registration — Project 2: Normalized low-rank adapters, the learning-rate band

Pre-registered before any GPU run. Commit hash recorded below at registration time.

**Registration commit:** `3743d4dce8d0949e0f45fbe6b3d3a41ddce7f2c6` (code + this document, before any GPU run)

## Question

Does normalizing an adapter's rank-one factors widen the range of learning rates at
which it trains successfully — and does per-component scaling add anything beyond a
single global scale? The accuracy delta is secondary.

## Arms (5, all implemented in `src/adapters.py`)

| arm | ΔW | what it isolates |
|---|---|---|
| `lora` | (α/r)·BA | baseline |
| `lora-scalar` | (α/r)·g·BA | gain, no normalization (control) |
| `lora-diag` | (α/r)·B·diag(s)·A | per-component gain, no normalization (control, missing from v1) |
| `delora` | (α/r)·g·ÛV̂ᵀ | normalization, one scalar (Bini et al. 2025) |
| `delora-pc` | (α/r)·Û·diag(s)·V̂ᵀ | normalization + per-component scale (thesis "DoLoRA") |

g, sᵢ = softplus(λ) − log 2, λ = 0 at init ⇒ ΔW = 0 at init for all arms. α = 2r.
(Thesis's `sᵢ ≥ 0` is wrong: range is (−log 2, ∞); we replicate verbatim.)

## Estimand (primary)

W_τ(arm) = log10 max{η : P(η) ≥ P*ₐᵣₘ − τ} − log10 min{·}, P* = arm's own
seed-mean sweep peak. Reported at τ = 1.0 accuracy point (absolute) AND τ = 2%
relative. Band on task accuracy is primary metric choice; band on val loss is
reported alongside.

**Contrasts:** normalization effect = W(DeLoRA) − W(LoRA); per-component delta =
W(DeLoRA-PC) − W(DeLoRA). DeLoRA-PC − LoRA is NOT primary (attributes published
normalization work to the untested part).

**Free exact null:** at r=1, DeLoRA-PC ≡ DeLoRA. Measured W difference at r=1 must
be 0; its spread is the noise floor. r=1 runs at every tier. Implementation-level
identity is unit-tested (`tests/test_identity_r1.py`).

## Grid and pinned protocol

- Common grid for EVERY arm: `logspace(1e-5, 1e-1, 13)` (1/3-decade spacing),
  including LRs where LoRA diverges — divergence is data.
- AdamW, weight_decay = 0, cosine schedule, warmup 2%, batch 32, no grad
  accumulation, α = 2r. Identical across arms.
- 3 seeds. CRN pairing where applicable.
- Band edges are grid-censored: W is a multiple of 1/3 decade; a one-point band
  (W = 0) means "width ≤ 1/3 decade".

## Divergence criterion (secondary estimand η_div)

A run is diverged if ANY of: (a) NaN/inf loss encountered; (b) mean train loss over
the last 10% of steps > loss at init; (c) max grad-norm > 1e4. Diverged runs count
as P = −∞ for band purposes. η_div = smallest grid LR where ≥ half of seeds diverge.

## Mandatory robustness check (Falsifier 1)

Re-express every band in effective-step units, e_t = ‖ΔW_{t+1} − ΔW_t‖_F / ‖W₀‖_F,
logged per step for every run. **If the arms' bands coincide on the effective-step
axis, the nominal-LR widening is a reparameterization artifact and the claim is
dead; the report must say so.**

## Pre-committed falsifiers

1. Bands coincide in effective-step units → artifact, claim dead.
2. `lora-diag` reproduces the widening → it is the gain parameterization, not the
   geometry.
3. W(DeLoRA-PC) − W(DeLoRA) inside the r=1 null → normalization widens, per-component
   adds nothing, finding belongs to Bini et al. (modal outcome; this sentence is the
   advance write-up).
4. Band predicts nothing downstream (cross-task LR-transfer regret null).

## Substrate and power

- **Vision grid (headline):** VTAB-19 × ranks {1,2,4,8,16,32} × 5 arms × 3 seeds,
  ViT-S/16, paired Wilcoxon. MDE 0.58 VTAB-mean points at ~43 GPU-h. Add FGVC
  10-shot. Keep the CV(‖W₀‖) diagnostic.
- **BoolQ collapse check (~8 GPU-h):** LoRA at r=2, tuned LR, 5-task concat at 4B,
  per-seed numbers + answer-prior distribution logged. Thesis BoolQ = 57.58 is below
  the ~62.2% majority baseline and contributes +5.62 of the +7.63 headline; with no
  SDs in the thesis, one collapsed seed is indistinguishable from a real effect.
- commonsense170k as a headline substrate: DROPPED (four simultaneous differences
  from the thesis make "replication" meaningless).

## Explicitly not run

- Appendix-A / singular-value-dispersion experiment. Theorem 1 is false (Eq. 8
  assumes mutually orthonormal ûᵢ; for a rank-2 target with σ = (10, 1) it predicts
  global-scale error 40.5 vs actual minimum 0.0). The free mutual-coherence
  diagnostic (max_{i≠j}|ûᵢᵀûⱼ|, logged every step) replaces it.
- AUC-under-LR-curve as primary (depends on swept range; reintroduces the
  disjoint-grid problem).

## Gate decision for Project 3

The vision-grid result is reported as a gate: PASS (normalized arms widen the band
beyond the r=1 null AND the widening survives the effective-step re-expression) /
ARTIFACT (bands coincide in effective-step units) / NULL (no widening beyond null).
Project 3 (RL tier) depends on this decision.
