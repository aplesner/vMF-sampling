# Project 3 — Normalized adapters under RLVR on code

**Read `00-shared-constraints.md` first. This is a separate paper from Project 2 — do not merge
them.**

## The question

Does the advantage of normalized per-component adapters **grow as the reward carries fewer bits**?

## Why separate from Project 2

Merging was proposed and rejected. Two reasons that stand:

- The merged primary estimand confounds normalization (published) with per-component scaling (the
  untested delta), and the budget for a five-condition dose axis comes out of the arm count — the
  DeLoRA control is what gets dropped to pay for it.
- The merged paper's escape hatch is not real: remove the RL half and you have deleted the spine and
  the title, leaving a language-only SFT design already labelled underpowered.

**Gate this project on Project 2's vision result.** If per-component scaling does not beat one global
scalar at 43 GPU-h with MDE 0.58, there is no novel method to test under RL.

## The dose axis — build it inside RL

The v1 axis (SFT → RFT → partial-credit RLVR → binary RLVR → binary at G=4) is **not one axis**. Its
levels differ in gradient estimator, data source, on/off-policy-ness, group-baseline variance, dead-
group fraction, and 6× in wall-clock. Specifically: **D1 and D2 are the same dose** (RFT is
rejection-sampling SFT — dense token CE, identical estimator, differing only in data source), and
**D4 vs D5 changes group-baseline quality, not signal bits**. The only genuine granularity step is
partial-credit vs binary, and one step is not a slope. "Bits per example" is also not well-defined
across CE and RL — for CE the informative content is the *surprisal*, which shrinks as the model
improves, making dose a function of training progress rather than of condition.

**Replace it with reward granularity at fixed everything else:** vary the number of visible tests
used to compute the pass-fraction reward, `k ∈ {1, 2, 4, 8, all}`, holding estimator, prompt set,
group size, token budget, optimizer step count, on-policy-ness and KL identical. Here "bits" is
defensible: a pass-fraction reward over k visible tests carries at most **log₂(k+1)** bits, versus
exactly 1 bit at k=1. Equally spaced in log, and only one factor moves.

**State the cost honestly:** the range shrinks from an illusory "dense CE → 1 bit" to a real
log₂(9) ≈ 3.17 → 1 bit, about 3×. Either the per-unit effect must be larger or more seeds are
needed. Do not let the wider fake range flatter the power calculation.

Keep the **G-sweep as a separate factor**, not folded into the dose scale — G varies baseline
variance and dead-group fraction, not signal bits.

## Power — get this right before spending

- The v1 MDE rests on an **assumed** CRN correlation ρ=0.7, which is a simulation number, not an RL
  number. At ρ=0.4 the MDE is 1.9 against a modal effect of +1…+2.5.
- **Run the ρ̂-estimation block first: 2 core arms × 3 seeds at 40% run length ≈ 15 GPU-h**, before
  committing the ~190 GPU-h main block. Write the response to ρ̂ < 0.5 now, not at step 300.
- In a within-seed dose design with a `(dose|seed)` random slope, **the effective n for the slope is
  the number of seeds**, not seeds × doses. Report SE(β̂) with seeds as the unit and MDE **per dose
  unit** on a compute-matched basis.
- 5 seeds at 4B, 3 at 9B. Recover resolution from CRN pairing and per-problem mixed effects, not
  seed count.

## Gradient SNR is a mediator, not the dose

Defining the dose axis by measured gradient SNR is circular and endogenous: SNR is the *mechanism's
dependent variable* (the hypothesis is that the tangent projection raises SNR under RLVR). It is also
arm-dependent — regressing `Δ = PC − LoRA` on SNR measured inside the LoRA arm is an endogenous
regressor, and seeds where LoRA happened to have poor SNR would mechanically show large Δ, giving a
**positive slope under the null**. Keep SNR where it belongs: logged per arm as a mediation variable
testing whether the experimenter-set knob works through the predicted channel.

## Data — code, not math

Decisive on two axes. **Partial credit only exists in code** — the graded-reward axis is simply
unavailable in math without step-level verification. And **post-cutoff math sets are 15–30 problems**
(AIME 2025/2026, HMMT), which cannot support the per-problem mixed-effects model the entire power
argument rests on, versus LiveCodeBench's ~500–700 date-stamped problems.

**Train:** PrimeIntellect/DeepCoder verifiable pool + TACO-verified, **filtered by measured base
pass@8 ∈ [0.15, 0.85]** on the frozen base — this *constructs* the 40–60% band instead of hoping for
it. Restrict to problems dated **≤ 2025-05-31**. Target ~4–8k problems, each with ≥8 tests (≥4 visible
for reward, ≥4 held out for scoring).

**Eval primary:** LiveCodeBench-v6, window **≥ 2025-06-01**, **all difficulties**. Not the easy
subset — that slice is both ceiling-compressed against a 70–88% base and the one most likely to
overlap training.

**This fixes a real train/test leak in v1**, which trains on LiveCodeBench-v6 and evaluates on
"LCB-v6 easy" with no stated disjointness. The temporal split is strictly better than dropping LCB
from training because it also supplies the contamination design.

**Eval secondary:** MBPP-Plus (378 problems, the per-problem analysis workhorse) and HumanEval+, both
labelled as saturated legacy anchors.

**Math as a zero-training transfer check (~2–3 GPU-h):** evaluate the code-trained adapters on
MATH-500 and AIME-2025/2026. Tests whether the effect is code-specific.

## Contamination protocol

Raw perplexity of canonical solutions **cannot** separate memorization from easiness — easy problems
have short, idiomatic, high-frequency solutions and therefore low perplexity with no memorization at
all. And residualizing on base pass rate conditions on a descendant of the outcome, inducing collider
bias in the analysis it is meant to protect.

Use instead, in order of value:
1. **Verbatim-memorization contrast:** base perplexity on the canonical solution vs a
   semantics-preserving syntactic perturbation (rename identifiers, reorder independent statements).
   Memorization produces a large gap; typicality and brevity do not. Difficulty-invariant by
   construction.
2. **Matched-null calibration:** pre-cutoff vs post-cutoff canonical solutions matched on base pass
   rate and solution token length.
3. Report contamination as a **pre-registered stratification**, not a continuous covariate.
4. State the assumed pretraining cutoff explicitly as an assumption.

The date-discontinuity RDD is underpowered and unidentified (the cutoff is not published, so the
threshold would be fitted to the data, and LCB difficulty drifts with contest composition). Demote it.

## Arms

LoRA / LoRA+diag(s) / DeLoRA / DeLoRA-PC, with CRN coupling and matched token budgets. **Primary
contrast is DeLoRA-PC − DeLoRA.** LoRA+diag(s) is the killer ablation: is the gain just a learnable
per-component gain?

**Do not pre-budget the RLOO immunization arm (~74 GPU-h).** The concern — that the effect is
specific to GRPO's z-scored group advantages — is legitimate, but the G-sweep already varies exactly
that quantity, and Project 2's vision tier can run `G ∈ {2,4,8,16}` for ~5 GPU-h. Gate RLOO on a
positive primary.

## Metrics

pass@1 (unbiased, n=16) with pass@16 as pre-registered co-primary, citing the RLVR-lowers-pass@k
result — but note pass@16 will be near-saturated on MBPP-Plus and lives only on the hard slice, which
is another reason LCB is primary. Diagnostics: gradient SNR, s=0 escape, dead-group fraction,
reward-vs-KL Pareto at β=0 via adapter-disable.

## Budget

~200 GPU-h at 4B expected after gates. Re-measure one real 9B GRPO step in week 1 — v1's 7.15 h/run
assumed a 1.14× multiplier over 4B that is not physical; the honest band is 1.8–2.4×, putting a 9B
tier near ~320 h rather than 200. Do not commit 9B before that measurement.
