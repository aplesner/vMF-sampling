# Project 2 — Normalized low-rank adapters: the learning-rate band

**Read `00-shared-constraints.md` first.**

## The question

Does normalizing an adapter's rank-one factors **widen the range of learning rates at which it
trains successfully** — and does per-component scaling add anything beyond a single global scale?

One question. The accuracy delta is a secondary result in this project, not the headline.

## Provenance

`papers/nLoRA.pdf` ("DoLoRA") is a **master's thesis supervised by the project owner** — never
published, never submitted, **code not retained**, so replication is a genuine prerequisite. It used
**SFT only, never RL** (RL is Project 3, and it is a separate paper).

Method: `ΔW = (α/r)·Σᵢ sᵢ·ûᵢv̂ᵢᵀ` with unit-normalized rank-one factors and per-component
`sᵢ = softplus(λᵢ) − log 2`. Prior art: **DeLoRA** (Bini et al., ICLR 2025) normalizes the same way
but with a **single per-layer scalar**. The one untested delta is per-component vs one scalar.

## Why the LR band is the headline

The thesis swept **disjoint** LR grids — LoRA/DoRA over {5e-4 … 1e-5}, DoLoRA over {5e-2 … 1e-3} —
because DoLoRA was empirically far more robust to high LRs while the baselines diverged. That was
**correct protocol**, not a confound: every selected LR is interior to its own grid at every rank on
both models, which is evidence each grid bracketed its optimum.

But the most interesting empirical finding in the thesis is buried as one unsupported sentence in a
protocol appendix, with no divergence criterion, no loss curves, and no count of diverged runs.
Promote it:

- **Immune to the undertuned-baseline attack by construction** — the estimand is referenced to each
  arm's *own* peak, so a stronger baseline makes the comparison fairer, not the claim weaker.
- **Survives an accuracy null**, which is the modal outcome (~25% prior on the accuracy delta
  landing outside the noise floor). Today that outcome is a null paper; under this framing it is a
  positive one.
- **Mechanism-level:** decoupling update direction from magnitude is exactly what should widen the
  stable range. The claim and the mechanism are the same object.
- **Nearly free** — the LR sweeps are already budgeted; you stop collapsing them to an argmax.

## Estimand

**Primary — band width in decades at a pre-registered tolerance τ:**

```
W_τ(arm) = log10 max{η : P(η) ≥ P*_arm − τ} − log10 min{η : P(η) ≥ P*_arm − τ}
```

with `P*_arm` that arm's own sweep peak. Fix τ in advance; report both τ = 1.0 accuracy point and
τ = 2% relative.

**Contrasts.** The per-component delta is `W_τ(DeLoRA-PC) − W_τ(DeLoRA)`. The normalization effect is
`W_τ(DeLoRA) − W_τ(LoRA)`. **Do not make `DeLoRA-PC − LoRA` primary** — it confounds normalization
(published, Bini et al.) with per-component scaling (the untested part), and a positive result on it
belongs to DeLoRA's authors.

**Free exact null:** at r=1, DeLoRA-PC and DeLoRA are the identical parameterization. The measured
band difference at r=1 *must* be zero, and its spread **is** the noise floor for the estimand. Run
r=1 at every tier.

**Secondary:** divergence threshold η_div (smallest η with NaN / loss above init / grad-norm
blow-up), criterion pre-registered. Do **not** make AUC-under-the-LR-curve primary — it depends on
the swept range, reintroducing the disjoint-grid problem.

**Zero-cost co-result — downstream utility.** Tune one LR on task A, apply it unchanged to tasks
B…K, measure accuracy regret vs per-task tuning. Converts "wide band" into "you can skip the sweep,"
which is the claim a practitioner acts on. Pure re-analysis of sweeps already run.

## Protocol change this forces

The thesis's disjoint grids were right for argmax selection and are **structurally incapable** of
supporting the band estimand — you cannot measure LoRA's upper band edge if you never ran LoRA above
5e-4. Use a **common, wide, dense grid for every arm**: {1e-5 … 1e-1} at 1/3-decade spacing
(13 points), **including the LRs where LoRA diverges** — a divergence is data now, not a wasted run.

Pin identically across arms and state them: schedule (cosine), warmup (2%), epochs, batch size.

## Confounds and the controls that separate them

- **Reparameterization, not conditioning — the number-one threat.** `sᵢ` initializes at 0 and gates
  the whole update, so the model can absorb a large nominal η by keeping `s` small. A 100–200×
  higher tolerable LR is *exactly* what a near-zero-initialized learnable gain predicts, with zero
  conditioning benefit. **Pre-registered primary robustness check: re-express both bands in
  effective-step units** — `‖ΔW_{t+1} − ΔW_t‖_F / ‖W₀‖_F` per step, logged for free. **If the bands
  coincide there, the widening is an artifact and the claim is dead.**
- **Gain vs normalization.** Run the full ladder: LoRA / LoRA + single learnable scalar /
  **LoRA + diag(s)** / DeLoRA / DeLoRA-PC. If LoRA+diag(s) widens the band as much, it is the gain
  parameterization, not the tangent projection. This arm is missing from v1's P2 entirely.
- **Metric choice.** Band on val loss ≠ band on task accuracy. Pre-register one, report both.

## Falsifiers, pre-committed

1. Bands coincide in effective-step units → artifact, claim dead.
2. LoRA+diag(s) reproduces the widening → it is the gain, not the geometry.
3. `W(PC) − W(DeLoRA)` inside the r=1 null → normalization widens the band, per-component adds
   nothing, **and the finding belongs to Bini et al.** This is the modal outcome; write it in advance.
4. The band predicts nothing downstream (regret test null).

## Substrate

**Vision is the headline tier, not a proxy.** VTAB-19 × ranks {1,2,4,8,16,32} × 5 methods × 3 seeds,
ViT-S/16, paired Wilcoxon: **MDE 0.58 VTAB-mean points at ~43 GPU-h**. This is the only tier in the
whole program that resolves the novel delta at the effect size it plausibly has (~0.5–1 pt). Reaching
MDE 0.58 in an RL tier would need ~29 seeds ≈ 360 GPU-h for a *single rank*.

Add FGVC 10-shot. Keep the free CV(‖W₀‖) diagnostic.

**Drop commonsense170k as a substrate.** Four simultaneous differences from the thesis make a
"replication" meaningless (base model, corpus, task set, scoring); the tasks are pre-2020 and
saturated; and the 8-task average has an across-task heterogeneity floor τ²/K that no seed count
removes — 2→5 seeds at 9B moves MDE only 1.30 → 1.06 for 104 GPU-h.

**Keep exactly one commonsense run (~8 GPU-h): reproduce the BoolQ collapse.** LoRA at r=2, tuned
LR, 5-task concat at 4B, **with per-seed numbers and the answer-prior distribution logged**. In the
thesis, BoolQ alone contributes **+5.62 of the +7.63** headline (73.6%), at a LoRA score of **57.58 —
below BoolQ's ~62.2% majority-class baseline**, while LoRA's own BoolQ recovers to 84.86 at higher
rank in the same table. So "the gain shrinks with rank" is arithmetically "LoRA's BoolQ recovers with
rank." No standard deviations appear anywhere in the thesis, so it is impossible to tell whether one
seed collapsed. If the collapse reproduces at a tuned LR, that is a real low-rank *stability* finding
that feeds the band story directly; if not, the headline was an artifact and you can stop citing it.

## Baselines

LoRA (tuned), DoRA, **AdaLoRA** (published per-component importance scaling, 2023, in PEFT — its
absence is the strongest novelty objection), DeLoRA, rsLoRA, LoRA+ (separate A/B learning rates — the
cheapest "it's all LR structure" rival). ~27 GPU-h for the missing ones.

## Do not run

**The Appendix A / singular-value-dispersion experiment (~15 GPU-h). Theorem 1 is false.** Its Eq. 8
assumes the ûᵢ are mutually orthonormal and nothing enforces that; for a rank-2 target with singular
values (10,1) the theorem predicts global-scale error 40.5 while the actual minimum is **0.0** —
exact fit, because γ·Σûᵢv̂ᵢᵀ with non-orthogonal unit factors realizes any spectrum with a+b ≤ 2.
Table 5's "spectrum constraint σ₁=…=σᵣ" is wrong. Use the free **mutual-coherence** diagnostic
(`max_{i≠j}|ûᵢᵀûⱼ|`) instead — near-orthogonality of learned factors is the empirical question that
actually determines whether per-component scales buy anything.

Also fix in passing: the thesis asserts `sᵢ ≥ 0` in three places; with `softplus(λ) − log 2` the range
is `(−log 2, ∞)`. The constraint is asymmetric (negative scales bounded by 0.693, positive unbounded),
so a sign-symmetric variant is the cleaner control for whether sign freedom matters.

## Budget

~110 GPU-h for the complete project. Vision grid 43, common-grid LR sweep ~12, missing baselines ~27,
BoolQ check ~8, slack.
