# Shared constraints — read before any project brief

Applies to all six projects. Where a project brief disagrees with this file, this file wins.

## Scope discipline

**One question per project. One question per paper.** Do not bundle a measurement contribution,
a method contribution, and an application into one write-up. If a project produces two clean
results, that is two papers, not one muddy one.

**Do not optimize for a venue.** Run the experiment, get the result, then decide where it goes.
Venue-first design is what produced the discarded v1 of this program.

## Compute

**Use the smallest GPU on which everything fits.** Escalate only when VRAM actually forces it, and
say in the run log what forced it.

| If it fits in | Use | Notes |
| --- | --- | --- |
| 24 GB | **2× RTX 3090** — the default | Two independent workers, one run per card |
| >24 GB | 1× A6000 (48 GB) | |
| >48 GB | A100 | Last resort; also the scale-demo tier |
| — | A few thousand H100-h (earned) | Only for established effects; confirmation, not discovery |

**Unless a brief says otherwise, you may use two RTX 3090s at once.** Prefer them as two independent
workers (one run per card) over data-parallel across both — nearly every tier in this program is a
many-small-runs design, where two independent streams give ~2× throughput while DDP on 2×3090 over
PCIe gives considerably less. Reach for DDP only when a single run genuinely does not fit in 24 GB.

Before escalating to a larger card, try the cheap fixes first: gradient checkpointing, gradient
accumulation instead of a larger batch, bf16, 8-bit optimizer states, and — for RL — disabling the
adapter to get the reference policy instead of holding a second copy of the base model (~16 GB saved
at 8B). A run that fits on a 3090 after twenty minutes of engineering is worth more than the same run
on an A100, because you can then have two of them at once.

Pre-training from scratch: stay ≤~100M params. **ImageNet-1k from scratch is affordable** with
FFCV (ResNet-18 ≈ 2–3 A6000-h/run, ResNet-50 ≈ 6–8). Measure images/sec on one epoch before
committing — published FFCV figures ran ~1.5× optimistic in our costing.

Post-training (SFT, RL, PEFT): bounded by 48 GB, and LoRA makes that generous (8B LoRA bf16
≈ 20–24 GB). For RL, the reference policy is free — disable the adapter rather than hold a
second copy.

Models trained or finetuned: **Qwen3.5** family (4B, 9B, 27B dense; 35B MoE). Projects that only
*measure* released checkpoints should keep generational diversity on purpose — Pythia's ~150
checkpoints have no modern replacement.

**Cost model correction:** the 4B→9B multiplier is **1.8–2.4×**, not 1.14×. The 1.14× figure in
v1 double-counted a fixed-overhead argument made for 1.5B→4B. Re-measure one real step in week 1
before committing any 9B tier.

## Storage

Two tiers, and using the wrong one is a common way to make a GPU-bound job IO-bound.

| Path | Property | Use for |
| --- | --- | --- |
| `/scratch/aplesner` | **Node-local.** Fast. Not visible from other nodes. | Anything read or written frequently: training data actually being consumed, FFCV/webdataset shards, checkpoints during a run, dataloader caches |
| `net_scratch` | Network. **Slower and IO-constrained**, but readable from anywhere. | Canonical storage: model weights, datasets of record, finished checkpoints, results to be aggregated across nodes |

**The working pattern:** keep weights and datasets on `net_scratch`; at job start, copy what the run
touches into `/scratch/aplesner` **if it is not already there**, and read from local for the duration.
Write hot artifacts locally, then sync the ones worth keeping back to `net_scratch`.

Practical notes:
- Guard the copy so repeated jobs on the same node do not re-transfer — check for the file first, and
  prefer an atomic pattern (copy to a temp name, then rename) so a killed job cannot leave a
  half-written file that later runs treat as complete.
- `/scratch` is node-local, so **do not assume anything survives** between jobs or is visible to a job
  landing on a different node. `net_scratch` is the only thing that persists across nodes.
- Do not run a many-small-files pipeline directly against `net_scratch` — that is the case its IO
  limits bite hardest. FFCV preprocessing in particular (~150–300 GB for ImageNet) must land on local
  scratch before the throughput figures in this document are achievable.

## Where your work lives

**Code goes in a top-level `projects/<your-project>/` directory** — e.g. `projects/continual/`,
`projects/adapter-lr-band/`, `projects/kappa-measure/`. Do not scatter code into the repo root or
into the library's own package.

> Note the two similarly-named locations, and do not confuse them:
> `docs/projects/` holds the **briefs** (read-only for you — this file and your project brief).
> `projects/` holds **your code and outputs** (yours to write).

`projects/` will be gitignored at the top level later. It is deliberately left tracked for now so
work is backed up — so keep large artifacts (checkpoints, datasets, raw logs) out of it and on
`net_scratch`, and commit only code, configs, and small result tables.

Suggested layout, not mandatory: `projects/<name>/{src,configs,results,report.html}`.

## Reporting

**Every project produces a self-contained HTML report at `projects/<your-project>/report.html`**,
updated as results land rather than written at the end. It is the primary channel for findings — the
project owner follows progress by opening it.

Requirements:
- **Self-contained.** Inline all CSS and JS; embed plots as data URIs or inline SVG. It must render
  correctly opened directly from disk with no network and no build step.
- **Inherit the existing design.** `docs/benchmark_report.html` and `docs/kappa_report.html` already
  establish this repo's report style; match their tokens so the set looks like one program.
- **Lead with status**, not methodology: what has run, what it showed, what is blocked, what is next.
  Put the gate decisions where they can be seen without scrolling.
- **Show the numbers.** Every claim gets its table or plot, with n, seeds, error bars, and the matched
  null where one applies. A claim without its uncertainty does not go in.
- **Record negatives and kill conditions as they fire.** A pre-registered condition that triggered is
  a result; log it prominently rather than burying it.
- Keep a short dated changelog at the bottom so the owner can see what is new since last time.

**Keep chat output minimal.** Findings belong in the report. Use the chat only for: blocking
questions, decisions that need the owner, a gate opening or closing, and anything that invalidates
another project's assumptions. Do not narrate routine progress, and do not paste result tables into
chat that already exist in the report.

## Measurement rules (any project touching κ)

- **SciPy is unusable at p ≥ 1024.** `ive` underflows to exactly zero above order ~511, so Sra's
  estimator returns NaN and safeguarded Newton silently stalls at ~1e-3. `ive(2048, κ)` returns
  NaN for κ ≲ 2048 — the admissibility formula itself is uncomputable in stock SciPy at p = 4096.
  Use this repo's CUSF/torch log-Bessel path.
- **Statistical error dominates numerical error by ~12 orders of magnitude** at high p. An accurate
  solver buys nothing if n is too small.
- **Every κ̂ ships with a matched null.** The finite-sample bias is systematically upward — the same
  direction as any "more concentrated than uniform" claim.
- **Do not use the rectangular rule** (`R̄ > 0.19 AND n/p ≥ 10`). It is grossly conservative in the
  high-concentration corner: at p=512, R̄=0.85, n/p=0.5 the measured relative RMSE is **0.63%**,
  better than the 1% target at *half* the samples the rule demands. At high concentration the bias
  floor is ≈1/n and does not involve p at all, so "n < p" carries no information once R̄ is large.
  Use the closed-form surface:

  ```
  rel-RMSE(κ̂) ≈ hypot( amp·(1−R²)/(2nR²),  1/(κ√(n·A_p′(κ))) )
  amp = 1 + 2R²/(1−R²) − 2R²/(p−R²)
  ```

  Matches simulation to 2 s.f. across the grid.
- **The bias shrinks gaps, it does not inflate them.** A common upward bias makes ordinal claims
  ("A is more concentrated than B") *conservative*: a true ratio of 2.0 reads as 1.267 at n=100.
  Ordinal comparisons at matched n survive; the real attack surface is comparisons at **unequal n**
  and low κ/p.

## Claims retired — do not reintroduce

These were asserted in v1 or in the extension proposals and were refuted numerically. Each is
listed with the correction so it does not get rediscovered.

| Retired claim | Correction |
| --- | --- |
| Known-mean estimation gives a "√p extension, 64× at p=4096" | Bias-vs-sd comparison. Detection threshold improves by **0.93·p^(1/4) ≈ 7.4×**; *estimation* accuracy gain is **exactly zero** (κ and μ are orthogonal parameters). With a split-sample μ it is **worse than plain Rayleigh**. |
| Per-class κ̂ at CIFAR-100 scale (n=500, p=512) is "unestimable" | Measured rel. RMSE **0.63%** — one of the best-conditioned cells in the program. |
| KappaFace-style per-class κ at n≈70, p=512 is an audit target | rel. RMSE 1.9–2.9% at κ/p ~1–3, Spearman(true, κ̂) = **0.995**. And κ̂ is **z-scored across classes** before use, so any common bias cancels by construction. Attacking a training-time weighting hyperparameter with measurement theory is a category error. |
| Isotropy instruments have "no null distribution" | Ethayarajh (2019) §3.4 defines and subtracts an anisotropy baseline, and its headline claims are ordinal. The narrow real crack: that baseline uses only 1K samples, so it carries its own sampling error. |
| S-VAE H2: "KL grows linearly in m at fixed κ" is false | The asymptotics are right (KL ≈ κ²/2m at κ≪m; Θ(m) at κ∝m) but **Davidson never operates there** — implied κ/m is 4.6e5, 527, 82, 18.0, **4.85** at d = 2,5,10,20,40. KL at fixed κ is **non-monotone in m**, peaking at m\* ≈ 0.76κ, and Davidson's whole sweep is on the rising branch. H2 is a fair description of the branch the data occupies. |
| "Informativeness requires κ = Θ(m)" | False. Rate R needs κ ≈ √(2mR) = o(m). A single scalar delivers any target rate at large m. |
| "Hyperspherical VAEs are structurally immune to posterior collapse" | **Fixed-κ is immune** (the rate term has zero gradient w.r.t. the encoder). **Learned-κ is not** (KL = 0 exactly at κ=0, with vanishing gradient). |
| DoLoRA Theorem 1 (per-component scaling necessary for rank-r expressivity) | **False.** Eq. 8 assumes the û_i are mutually orthonormal; nothing enforces that. For a rank-2 target with singular values (10,1), Theorem 1 predicts global-scale error 40.5; the actual minimum is **0.0**. Do not spend GPU-hours testing its prediction. |
| The DoLoRA disjoint LR grids were a confound | **Withdrawn.** Every selected LR is interior to its own grid at every rank on both models — evidence the grids bracketed their optima. Per-arm sweeping over arm-appropriate ranges is the correct protocol. |

## Statistical practice

- **3–5 seeds**, not 12. Recover resolution from CRN pairing, per-item paired analysis, and pooling
  across ranks/tasks — not from seed count.
- **Every arm gets its own LR sweep.** No arm is ever compared at a shared LR.
- **State an MDE before running.** If the design cannot detect the modal predicted effect, change
  the design or the estimand — do not run it and hope.
- **Random slopes do not drop out.** In a within-seed dose design with `(dose|seed)`, the effective
  n for the slope is the number of **seeds**, not seeds × doses.
- Pre-register the success criterion with a commit hash before the run.

## Shared dependency

Project 6 ships `vmf-measure` (null quantiles, admissibility surface, refusal mode, log-Bessel
path that works at p ≥ 1024). Projects 1 and 5 need it in week 1. **Freeze the API and table
schema by end of week 2 with a named owner**; numerical values may move afterward. Dependents pin
the schema, not the numbers.

**Unresolved, needs a decision before any double-blind submission:** several projects version-pin
the same public artifact. Three concurrent double-blind submissions pinning one public repo
de-anonymizes the group and reads as salami to any reviewer who checks. Anonymized release breaks
the pin-and-cite workflow. No good answer yet.
