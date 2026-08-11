# Research program: representations on the hypersphere

A plan for six papers built on constraining neural representations to the unit sphere, measuring
what that does with directional statistics, and exploiting it where the signal is scarce —
continual learning, low-rank adaptation, reinforcement learning, and adversarial robustness.

**Status of this document.** This section is written from the repository's own measurements and the
two seed results the group already has. Sections 2–7 were drafted against literature briefs and
should be treated as proposals to argue with, not commitments. Section 8 is an adversarial review of
sections 2–7, written without seeing this overview. Where they disagree, section 8 is usually right,
because it was asked to be hostile and the section authors were asked to be constructive.

---

## Read this first

Four findings from drafting that change what to do next week. They are buried in a long document
otherwise.

**1. DoLoRA is largely a rediscovery of DeLoRA — and this makes P2 a better paper, not a dead one.**
[DeLoRA](https://arxiv.org/abs/2503.18225) (Bini, Girrbach, Akata, ICLR 2025) already normalizes the
low-rank update, decouples angular learning from adaptation strength, and is
[merged into HuggingFace PEFT](https://huggingface.co/docs/peft/en/package_reference/delora). The
`papers/nLoRA.pdf` submission does not cite it. The remaining delta is narrow but real: DeLoRA
learns **one boundary scalar λ per layer**; DoLoRA learns **per-component scales `s_i`**, one per
rank-1 factor. And DoLoRA's own Appendix A proves that a *global* scale is a strictly worse
approximator than per-component scaling whenever the target's singular values are non-uniform —
which is exactly the theoretical case against DeLoRA, stated by a paper that never ran the
comparison. So P2's real question is not "does normalizing LoRA help" (answered, published) but
**"does per-component strength beat a single scalar, as the theory predicts?"** That is a sharper,
cheaper, more defensible paper, the baseline is a `pip install` away, and it should be
pre-registered as a diagnostics-and-rigor paper rather than a method paper.

**2. The +7.63 headline is unlikely to survive a tuned baseline.** The section-3 author puts ~55%
on the r=2 gain collapsing to +1…+2.5 once the LoRA baseline gets its own LR sweep and rsLoRA
scaling. Plan for that outcome rather than being surprised by it.

**3. P3 as scoped cannot reject its own null.** Its claim is an *interaction* — that the normalized
adapter helps RL more than it helps SFT — with a minimum detectable effect of ~5.5 points at 3
seeds, while the SFT gap it is measured against may itself be only +1…+2.5. That is a ~12,600
H100-hour commitment to an unfalsifiable question. The fix is to buy seeds with ranks: restrict to
r ∈ {1,2} and run 8 seeds.

**4. P6's core novelty claim is largely prior art.** I pitched the neural-collapse hook — that κ̂ is a
better-specified NC1 metric — as P6's strongest idea. The second-pass review found that per-class vMF
fitting of classifier features with an *estimated* κ is already SIREN (NeurIPS 2022), KappaFace,
CIDER, and the vMF-mixture-softmax line, and that directional NC1 is HUG/GNC. What remains unclaimed
is narrower: the **null-calibrated admissibility boundary** and the **p-sweep**. That is a methods
contribution, not a CVPR paper. Either narrow P6 to the measurement contribution and target a methods
venue, or find a genuinely new claim before spending its 99 hours. The detection half separately faces
Carlini & Wagner (2017) and Tramèr (ICML 2022) and currently has OOD-era baselines, no corruption
control, and no held-out attack.

**5. Compute is now internally inconsistent by ~5%, not 3–16×.** The earlier ~20,900-hour incoherence
is resolved. What remains: P2 and P3 price the same 4B→9B jump at 2.4× and 1.14× respectively, and
P1's FFCV numbers are ~1.5× optimistic against its own A100→A6000 conversion. Real total is
~1,900–1,980 across two card types. Worth reconciling, but it changes no decision.

---

---

## Contents

| § | Section | Venue | Tier 0 | Full |
| --- | --- | --- | --- | --- |
| 1 | Overview: thesis, seeds, constraints | — | — | — |
| 2 | **Hardware and model envelope** — authoritative, supersedes §3–8 | — | — | — |
| 3 | **P1** — Hyperspherical networks for class-incremental continual learning | CVPR | 10.0 h | 400 h |
| 4 | **P2** — Where do low-rank adapter gains come from? | ICLR | 7.9 h | 571 h |
| 5 | **P3** — Conditioning, not capacity: normalized adapters for RLVR on code | ICLR | 6.8 h | 656 h |
| 6 | **P4** — Measuring representation geometry with vMF concentration | ICLR | 4.0 h | 72 h |
| 7 | **P5** — Breaking the dimension wall in hyperspherical VAEs | ICLR | 5.0 h | 97 h |
| 8 | **P6** — Classifier latents and adversarial robustness on the sphere | ⚠ see §11 | 8.0 h | 99 h |
| 9 | Cross-cutting: infrastructure, sequencing, portfolio | — | — | — |
| 10 | **Compute tiers and the escalation ladder** | — | — | — |
| 11 | **Adversarial program review (second pass)** — read before starting anything | — | — | — |

**~1,900 GPU-hours total; every gate opens for ≈42 GPU-hours.** That second number is the one to act
on: roughly two days of compute decides which of six projects are real, and expected spend is far
below the committed total because most gates are designed to fail cheaply. The review in §11 argues
the true total is **~1,900–1,980 across two card types**, because P2 and P3 price the same 4B→9B jump
differently (2.4× vs 1.14×) and P1's FFCV figures are ~1.5× optimistic against its own A100→A6000
conversion. Reconcile before committing; it does not change any ordering.

Sections 3–11 were drafted in parallel by separate agents against shared constraints and independent
literature briefs, then re-scoped twice as the compute envelope firmed up. They occasionally disagree
with each other; section 11 catalogues where, and those disagreements are left visible rather than
smoothed over, because each marks a real decision the group has not yet made. Where a project section
names stale hardware or a stale model, **section 2 wins**.

---

## The thesis, stated so it can be attacked

> Constraining representations and weights to the unit sphere removes the scale degree of freedom
> from optimization. Where the learning signal is dense and plentiful, this buys little — the
> network finds a good scale anyway. Where the signal is **scarce, noisy, or non-stationary**, the
> removed degree of freedom is one fewer thing to get wrong, and the constraint pays. Directional
> statistics is the right instrument to measure this, because on the sphere the only remaining
> quantity *is* direction.

Three of the five projects are direct tests of the second sentence, in three different regimes of
scarcity:

| Regime of scarcity | Project | Prediction |
| --- | --- | --- |
| Non-stationary signal — the task distribution changes under you | P1, continual learning | Less forgetting, faster adaptation |
| Low-capacity signal — very few trainable parameters | P2, DoLoRA at low rank | Gains grow as rank falls |
| Sparse, high-variance signal — binary verifiable rewards | P3, RL on code | Gains exceed the SFT gains, and seed variance falls |
| Adversarially perturbed signal | P6, robustness | Angular perturbation is the right threat geometry; normalized nets resist it better |

The remaining projects supply the instrument (P4) and test the same geometry in a generative
setting where the sphere is a modelling choice rather than an optimization one (P5).

P6 also carries the program's most favorable measurement regime. Elsewhere κ̂ is fighting the
feasibility limits below; in a trained classifier, within-class latents are *highly* concentrated —
that is exactly what neural collapse NC1 asserts — so κ is large, κ/p is comfortable, and the
estimator works as intended. That makes P6 both a robustness paper and the cleanest demonstration
that the instrument measures something real. The connection is worth stating plainly: **NC1 is
conventionally a ratio of within- to between-class covariance traces; on normalized features the
principled directional version of within-class variability is the vMF concentration.** κ̂ is a
better-specified NC1 metric — dimension-comparable, likelihood-based, and equipped with a null.

**The honest caveat, up front.** This thesis has a well-known competing explanation: normalizing
weights changes the *effective learning rate*, and a good deal of what looks like "geometry helps"
in the literature has turned out to be "you accidentally tuned the step size better." That confound
threatens P1, P2, and P3 simultaneously. It is the single most important thing this program must
control for, and the reason section 7 specifies one shared protocol for it rather than leaving each
project to improvise. If the effect does not survive matched-effective-LR controls, the program's
thesis is wrong, and finding that out cheaply is worth more than three papers built on sand.

---

## Where this starts from

Two unpublished positive results and one finished piece of infrastructure.

### Seed 1 — continual learning (needs replication; code not retained)

An MLP trained on 5 of CIFAR-10's classes to convergence, then its head extended by 5 rows and
trained on the 5 held-out classes with no rehearsal. With nGPT-style normalized MLPs the model
converged **faster on the new classes** *and* **forgot less on the old ones** — which is notable
because those usually trade off against each other. A 3D PCA projection showed the normalized
model's embeddings barely moving during phase 2 while the unnormalized model twisted the whole
space.

That last observation is the seed of the whole program's measurement story, and it is currently a
picture. Turning it into a number with a null is P1's methodological contribution and P4's reason
to exist.

### Seed 2 — DoLoRA (needs replication; code not retained)

`papers/nLoRA.pdf`, an anonymous ICLR 2026 submission, normalizes LoRA's rank-1 factors directly
and learns per-component non-negative scales:

```
ΔW = (α/r) · Σᵢ sᵢ · ûᵢ v̂ᵢᵀ ,   ‖ûᵢ‖ = ‖v̂ᵢ‖ = 1 ,   sᵢ = softplus(λᵢ) − log 2 ,   λᵢ init 0
```

Reported gains are largest at low rank — LLaMA-3-8B at r=2 goes from 75.42 to 83.05 average, GSM8K
from 53.5 to 61.0 — and narrow to about +1 by r=32. The group reproduced better convergence and
accuracy on SFT.

Two details make this more than a PEFT trick. First, its gradient
`∇_{bᵢ} ∝ (sᵢ/‖bᵢ‖)(I − b̂ᵢb̂ᵢᵀ)(∇_W L) âᵢ` is an **orthogonal projection onto the tangent space of
the unit-norm constraint** — the same Riemannian retraction structure as nGPT's block update, which
is what makes P2 and P3 part of this program rather than a detour. Second, the paper's own stated
limitations are precisely two of the proposed projects: *"other modalities, such as Vision
Transformers"* and *"the interaction between component-level decomposition and specialized training
objectives like RLHF remains an area for future work."*

### Infrastructure already built

- **Fast batched vMF sampling** — 126.97× (NumPy) and 99.01× (Torch) over the SciPy reference at
  d=4096, single Householder reflection, device-native, fp16 through fp64.
- **Batched κ estimators** — Banerjee, Sra's two-step, safeguarded Newton, against compiled CUSF
  log-Bessel; 5.3M estimates/s at batch 2²⁰.
- **A characterized estimator**, which is what makes it usable rather than merely fast. See below.

---

## The constraint that shapes every measurement in this program

The κ benchmark (`docs/kappa_report.html`) established three facts that bound what the measurement
projects can claim. They are repeated in each section because ignoring them produces confident
nonsense.

**SciPy's Bessel path is unusable at p ≥ 1024.** `scipy.special.ive` underflows to exactly zero
above order ≈511. Sra-with-SciPy returns NaN; safeguarded Newton *silently* stalls at ~1e-3
relative error — the dangerous failure, because it returns a plausible number. Every dimension this
program cares about (768, 1024, 4096) is past that line, so CUSF or the PyTorch port is mandatory,
not preferred.

**κ/p governs accuracy, not p.** Above κ/p ≈ 0.2, roughly n = 10–25p samples buy 1% accuracy. Below
κ/p ≈ 0.05, even n = 10⁵ does not. A high-dimensional but concentrated fit is easy; a
low-dimensional diffuse one is hard.

**The finite-sample bias is systematically upward, and it points the same way as the claim.**
Anyone fitting a vMF to representations wants to conclude "more concentrated than uniform." That is
exactly the direction the artifact pushes. So a **matched null** — n uniform unit vectors in ℝᵖ,
same estimator, same n, whose mean is *not* zero — is not good practice, it is the only thing
separating the measurement from the artifact.

Practical consequence for this program: κ̂ on a handful of vectors is hopeless. Any proposed
measurement must state its (n, p, κ) regime and show it lands in the estimable region, or use a
different instrument. Where κ̂ does not apply, CKA, Procrustes disparity, and subspace principal
angles do, and every geometric claim in this program should rest on more than one of them.

---

## The six projects at a glance

| # | Project | Venue | What is new | Biggest risk |
| --- | --- | --- | --- | --- |
| P1 | Normalized networks for continual learning | CVPR | Full hyperspherical architecture + quantified drift, beyond known cosine-classifier results | Prior art in cosine classifiers; effective-LR confound |
| P2 | DoLoRA replication and extension to vision | ICLR | Reimplementation, mechanism diagnosis, the modality gap the paper names | The +7.63 at r=2 may not reproduce |
| P3 | Normalized adapters for RL on code generation | ICLR | Tests the paper's own stated future work; sparse-reward regime | RL variance may swallow the effect |
| P4 | Directional statistics as a measurement instrument | ICLR | Bias-corrected κ̂ in the LLM regime; null tables; audit of the isotropy literature | A measurement paper needs a surprising finding |
| P5 | Hyperspherical latent models at scale | ICLR | Breaks the m>40 wall; honest vMF vs Power Spherical | May confirm the competitor rather than beat it |
| P6 | Classifier latents and adversarial robustness on the sphere | CVPR | κ̂ as a directional NC1 metric; vMF noise at matched angular displacement; angular certificates | Gradient masking could fake the robustness result |

P4 is the instrument the others use. P1 is the flagship because it already has a positive
preliminary result; P6 is the safest bet because its measurement half is nearly guaranteed to
produce something publishable even if the robustness half fails. A defensible sequencing argument
is in section 7, and section 8 argues with all of this.

**One warning that belongs at the top.** P6's robustness claims and P1's stability claims are both
vulnerable to a defense that is not real: normalization can break gradient-based attacks without
conferring robustness (obfuscated gradients), just as it can improve continual learning by
accidentally rescaling the learning rate. Both projects must run their respective control suites —
transfer and black-box attacks for P6, matched-effective-LR for P1 — before any claim is written
down. A result that survives those controls is worth far more than three that do not.


---

## Hardware and model envelope

**This section is authoritative and supersedes any hardware or model assumption made in sections
2–7.** Those sections were drafted across several revisions of the budget; where one of them names a
different GPU, a different model, or a stale cost, this section wins.

### Compute

| Tier | Hardware | Rule |
| --- | --- | --- |
| Default | 1× RTX 3090 (24 GB) or 1× A6000 (48 GB) | ≤1000 GPU-h per project; target the lowest decisive tier |
| Scale-up | 2× A100 | Available, but deliberately **later** — to demonstrate scale once an effect is established, not for exploration |
| Earned | A few thousand H100-h | High-belief projects only; see the escalation ladder |

The distinction that matters is **pre-training versus post-training**, not raw model size.

- **Pre-training from scratch:** stay small, ≤~100M parameters — nanoGPT/nViT/nMLP ladders. The one
  correction worth stating loudly: **ImageNet-1k from scratch is affordable** with a fast data
  pipeline. FFCV reaches 75% top-1 with ResNet-50 in ~2.7 A100-h (≈6–8 A6000-h per run), and
  ResNet-18 at ~3× the throughput is ~2–3 A6000-h per run. It requires preprocessing to FFCV format
  on fast local NVMe (~150–300 GB); measure images/sec on one epoch before committing. ViT from
  scratch on ImageNet remains out (~300–500 A100-h/run for DeiT-S).
- **Post-training** (SFT, RL, PEFT, inference): bounded by 48 GB, and **LoRA makes that generous**,
  since the frozen base carries no gradients and no optimizer states. 8B LoRA bf16 ≈ 20–24 GB;
  14B ≈ 32–36 GB; 32B via QLoRA (NF4) ≈ 18 GB. Full finetuning caps near 1.5B. For RL, the reference
  policy is free — disable the adapter rather than holding a second copy (~16 GB saved at 8B).

**Consequence: VRAM is rarely the binding constraint. Wall-clock is.** Cost every experiment in
hours, not gigabytes.

### Models

Anything **trained or finetuned** uses the **Qwen3.5** family (Apache 2.0): **4B, 9B, and 27B dense**
plus a **35B MoE**. Do not use older Qwen generations.

This applies to training, not to analysis. Projects that *measure* released checkpoints (P4's
embedding atlas, P6's deferred LM study) should keep generational and architectural diversity on
purpose — Pythia's ~150-checkpoint training trajectory in particular has no modern replacement and is
the only way to measure how geometry evolves over training.

### A note on scope discipline

The compute planning in this document went through three revisions and consumed more attention than
it deserves. It is now settled; the numbers below are good enough to start on. The gates matter more
than the estimates — every project's Tier 0 is a handful of GPU-hours, and running one of them will
teach more than another round of costing. **Other promising directions exist outside this program and
should not be crowded out by it.**


---

## Project 1 — Hyperspherical Networks for Class-Incremental Learning: Architecture-Wide Normalization and a Null-Calibrated Measurement of Representation Drift

> **COMPUTE ENVELOPE (binding).** Total re-scoped budget: **≈ 400 RTX-A6000-hours** (48 GB), across **≈ 1,480 training runs**. Hard ceiling 1,000 GPU-h. On a single RTX 3090 (24 GB) the same program costs **≈ 490 GPU-hours** — still under the ceiling, but with two arms (ResNet-50/ImageNet at batch 256, and any full ViT-B finetune) pushed to the edge of or past 24 GB; see §1.7.3. This supersedes a previous re-scope that came in at 98 3090-hours by **deleting ImageNet-1k and the entire pretrained-model tier**. That cut was wrong. It assumed 90-epoch torchvision recipes and network-storage JPEG decoding, and on those assumptions ImageNet is indeed unaffordable. With **FFCV** (or DALI/WebDataset) plus a modern short-schedule recipe, class-incremental ImageNet-1k with ResNet-18/50 costs ~214 GPU-h for the whole study, and the pretrained-ViT tier — whose per-run cost was never the problem — costs ~89 GPU-h once its *breadth* is trimmed from 9 baselines × 2 backbones × 3 benchmarks × 5 seeds to 5 baselines × 1 backbone × 2 benchmarks × 3–5 seeds. Both are back in scope. The venue target is **CVPR main track again** (§ "Venue position").

**Claim.** Constraining *every* representation and *every* weight row of a classifier to the unit sphere (nGPT-style: normalized hidden states, normalized weight rows, learnable per-coordinate residual "eigen learning rates" α, cosine output head with learnable logit scale) improves the **stability–plasticity frontier** in rehearsal-free class-incremental learning (CIL) — not merely the endpoint — and it does so because it suppresses **coherent representation drift**: old-class features rotate less, and what rotation remains is closer to a rigid, label-preserving motion.

**Why it should be true.** Forgetting in CIL is largely (i) output-head magnitude/bias imbalance between old and new classes (LUCIR, CVPR 2019; BiC, CVPR 2019; Weight Aligning, CVPR 2020) and (ii) drift of old-class prototypes in feature space (SDC 2004.00440; LDC 2407.08536), with the backbone degrading far less than task accuracy suggests (Davari et al., 2203.13381). Cosine heads fix (i) by construction. Full-network normalization additionally removes the uncontrolled weight/activation magnitude growth that Karras et al. (EDM2, 2312.02696) identify as the actual pathology, and pins each layer's *angular* update size to an equilibrium (Kosson, 2305.17212; Wan, 2006.08419) so that phase-2 gradients cannot silently acquire a large effective step on the phase-1 solution. The mechanism is therefore "bounded angular motion per step" — a *measurable* prediction, not just an accuracy claim.

**Pitch (one sentence).** We take nGPT's hyperspherical architecture out of language modeling and into class-incremental learning, evaluate it from CIFAR-10 MLPs all the way to from-scratch ImageNet-1k and the pretrained-ViT SOTA cluster, sweep every confound rather than arguing it away, and replace the field's eyeball-level "the representation moved less" claim with a null-calibrated von Mises–Fisher concentration measurement of per-sample angular drift, validated in the (n, p) regime where that estimator is trustworthy.

**Honest prior-art position — read before writing the intro.** "Cosine/normalized classifiers reduce forgetting" is **already published**, repeatedly: Gidaris & Komodakis (CVPR 2018), Qi et al. imprinted weights (CVPR 2018), LUCIR (CVPR 2019), PODNet (ECCV 2020), Lesort et al.'s last-layer study (2106.01834). "Fixed hyperspherical/simplex-ETF prototypes reduce forgetting" is also published (NC-FSCIL, ICLR 2023, 2302.03004; ProNC, 2505.24254; NCCL, 2412.02865; hyperspherical prototypes, Mettes et al., NeurIPS 2019). "vMF distributions in continual learning" is published: **Michel et al., AAAI 2024** (*Learning Representations on the Unit Sphere*) fit a mixture of vMF with per-class prototype and concentration for *online* CL — cite and differentiate **on page 1**. "Hyperspherical normalization preserves plasticity" is published on the RL side (NaP, 2407.01800; SimbaV2, 2502.15280). The defensible novelty budget is exactly two items: **(a)** architecture-wide nGPT normalization (weights + hidden states + eigen-LRs, LayerNorm removed) in CIL — nobody has done nGPT-in-CL, and it is *not* the same as a cosine classifier; **(b)** treating the geometry claim as a **measurement problem** with a matched null, a stated feasibility regime, and numerically valid log-Bessel at p ≥ 1024 (and now p = 2048, where the problem is worse). Item (b) is the safer contribution and should carry the paper if (a) shrinks under ablation. The scale (ImageNet-1k + PTM SOTA) is not itself a contribution; it is the *admission ticket*, and the paper must not pretend otherwise.

**Venue position: CVPR main track, and here is the argument.** Three things a CVPR reviewer in this area demands, and how the restored plan supplies each.
1. **From-scratch large-scale CIL.** Split-ImageNet-1k with ResNet-18 *and* ResNet-50, 10×100 B0 plus the B500+5×100 large-base convention that the LUCIR/PODNet/iCaRL ImageNet tables actually use, 2 orders × 3 seeds (§1.4, Tier 3). This is the axis whose absence forced the previous downgrade.
2. **Comparison against the current SOTA cluster.** Split-ImageNet-R 10×20 and PTM Split-CIFAR-100 10×10 with a pretrained ViT-B/16 against **L2P, DualPrompt, CODA-Prompt, RanPAC, EASE** plus a **SimpleCIL** frozen-feature control, 5 seeds (§1.5, Tier 4). These are run at their **published full schedules** — they are cheap — so the SOTA comparison is at full fidelity even though the from-scratch ImageNet arm is not.
3. **A contribution that is not "we tried X and it went up."** The null-calibrated directional-drift measurement, the κ/p feasibility map, the SciPy `ive` underflow finding, and the n < p per-class collapse (§1.2). This is what distinguishes the paper from the twelfth normalization-helps-forgetting submission.

The one honest asterisk: the from-scratch ImageNet numbers use short schedules (16–32 epochs/task-equivalent, ResNet-18 ≈ 67–71% joint top-1 rather than 76%). Those numbers support **relative** comparisons between arms, all trained under an identical budget, and the paper must say so in the caption of every ImageNet table. Absolute-accuracy SOTA on from-scratch ImageNet CIL is explicitly **not** claimed and is listed as deferred (D3). A CVPR reviewer will accept a budget-matched relative comparison that is stated as such; they will not accept a budget-matched comparison dressed up as SOTA.

**What came back into scope relative to the previous version.**
- **Split-ImageNet-1k CIL from scratch**, ResNet-18 (primary) and ResNet-50 (confirmation), two protocols, joint oracles included. Previously "every ImageNet-1k-scale operation" was cut. (+214 GPU-h)
- **The pretrained-ViT / Split-ImageNet-R / PTM-SOTA tier**, trimmed to 5 baselines + 1 control + 2 of our arms, one backbone (ViT-B/16, ImageNet-21k pretrained), 2 benchmarks, 3–5 seeds. Previously deleted in full. (+89 GPU-h)
- **Wider ResNet-18 breadth on Tiny-ImageNet** (4 methods × 2 orders × 3 seeds rather than 3 × 2 × 2), because Tiny-ImageNet is now the Gate-2→3 predictor for ImageNet and needs to be trustworthy.

**What remains deferred, and what it would cost.** nViT-full with LayerNorm actually removed (needs ImageNet-scale re-adaptation of the destroyed checkpoint); from-scratch normalized-ViT/DeiT-S ImageNet-1k pretraining; full-fidelity 88-epoch ImageNet CIL for absolute-accuracy claims; the sequential-language-domain experiment; CORe50/OmniBenchmark; and the PTM baselines we deliberately dropped (SLCA, APER, MOS, CL-LoRA). Full accounting with H100 asks in §1.9.

---

### 1.0 The cheap-first ladder and its decision gates

Everything below is organized as five tiers with four gates. **Tier 0 answers the go/no-go for 10 GPU-hours** and remains the thing that decides whether anything else runs. Nothing in tier k+1 starts until gate k→k+1 passes. If a gate fails, the program pivots to the measurement paper (§1.8, "Tier M"), which costs ~4 GPU-h because it reuses checkpoints that already exist.

| Tier | Content | GPU-h (A6000) | Cumulative |
|---|---|---|---|
| **Tier 0** | CIFAR-10, MLP vs nMLP, full per-arm (η, λ) sweep, 25-run headline, full stability–plasticity frontier, complete §1.2 drift measurement | **10.0** | 10.0 |
| **Tier 1** | Component decomposition (9 variants), width × depth grid, canonical 5×2 protocol, 8 rehearsal-free baselines — all still MLP-scale | **30.0** | 40.0 |
| **Tier 2** | ResNet-18 and ViT-S/16-retrofit × Split-CIFAR-10 / Split-CIFAR-100 / Split-Tiny-ImageNet | **57.0** | 97.0 |
| **Tier 3** | **Split-ImageNet-1k from scratch**, ResNet-18 + ResNet-50, FFCV short-schedule | **214.0** | 311.0 |
| **Tier 4** | **PTM tier**: pretrained ViT-B/16, Split-ImageNet-R + PTM Split-CIFAR-100, 5 SOTA baselines | **89.0** | **400.0** |

**Gate 0 → 1 (evaluated after T0.5).** Comparisons are paired over the 25 (class-partition, seed) cells, each arm at *its own* tuned (η, λ).
- **GO** requires **both**: (i) median paired difference in final old-class Class-IL accuracy, nMLP − best-tuned-MLP, **≥ 3.0 points with a 95% bootstrap CI excluding 0**; and (ii) **frontier dominance** — the nMLP's (new-class acc, old-class acc) trace lies up-and-to-the-right of the baseline's over ≥ 60% of the phase-2 epoch grid, i.e. old-class accuracy **at matched new-class accuracy** is higher (§1.6, metric `A@matched-P`).
- **STOP** if the median paired difference is ≤ 1.5 points, or its CI includes 0, or **the two frontier traces overlap within noise**. Overlapping traces mean the arms sit at different *points on a shared frontier*, and the endpoint difference is an artifact of where phase-2 training happened to stop. That is a negative result and it is cheap to reach. Pivot to Tier M.
- The ambiguous band (1.5–3.0 points, or dominance on 40–60% of the grid) buys **Tier 1 only**.

**Gate 1 → 2 (evaluated after T1.7).**
- **GO** requires **both**: (i) the residual of full-nMLP over **cosine-head-only (LUCIR-lite)**, each at its own tuned LR, is **≥ 2.0 points median paired with a CI excluding 0**; and (ii) the drift separation is **null-calibrated and co-varying**: |z| ≥ 3 against the tangent-space matched null (§1.2), *and* Spearman ρ ≥ 0.5 between the per-cell R̄ gap and the per-cell accuracy gap across the 18 (width × depth × arm) cells. Co-variation is what turns a correlation into a mechanism claim; the width/depth grid exists to supply enough cells to test it.
- **STOP** if cosine-head-only accounts for ≥ 80% of the gap and the residual CI includes 0. Then the accuracy claim is LUCIR rediscovered, the paper is the measurement paper, and Tiers 2–4 are not spent.

**Gate 2 → 3 (the ImageNet gate). Two independent conditions, and the infrastructure one is checked FIRST and separately.**
- **Gate 2→3a — infrastructure, ~4 GPU-h, run before any science.** Build the FFCV `.beton` (or DALI/WebDataset shard set) and **measure sustained images/sec over one full ResNet-18 training epoch from local NVMe**. Requirement: **≥ 3,500 img/s sustained** on the A6000 (≈ 6 min per ImageNet epoch) with GPU utilization ≥ 85%. **If throughput is < 2,000 img/s, you are I/O bound and Tier 3 does not happen at this budget** — the whole cost model in §1.7.4 collapses, and the correct response is to fix the storage path or drop back to the Tier-0–2 + Tier-4 program (186 GPU-h, ICLR/TMLR). This is a hard prerequisite, not a nicety; see §1.7.4 for the storage requirement.
- **Gate 2→3b — science.** Tier 2 must show the effect surviving to (i) ResNet-18 from scratch, (ii) a 10-task sequence, and (iii) Split-Tiny-ImageNet specifically: median paired advantage over the best-tuned cosine-head-only baseline **≥ 2.0 points final average accuracy A_T with CI excluding 0**, on ≥ 2 task orders. Tiny-ImageNet is the closest cheap proxy to ImageNet (200 classes, 64², 10 tasks) and is deliberately the predictor.

**Gate 2 → 4 (the PTM gate; independent of Gate 2→3 and evaluatable in parallel).** Unlock Tier 4 if the **exact, function-preserving ViT-S/16 retrofit** (T2.7) beats cosine-head-only by ≥ 2.0 points on Split-CIFAR-100 10×10 across ≥ 2 orders, **or** shows a null-calibrated |z| ≥ 3 drift separation at p = 384 even without an accuracy win. The `or` is deliberate: the PTM tier is where the *measurement* contribution is most valuable to the community (every PTM-CIL paper makes uncalibrated representation-stability claims), so a measurement-only result still justifies spending it. If both fail, Tier 4 is dropped and the paper is Tier 0–3 — still a real paper, and still ImageNet-scale.

---

### 1.1 Replication (the exact prior experiment, made rigorous) — Tier 0

The original result (CIFAR-10, MLP vs nMLP, 5 classes then 5 classes, head extended by 5 rows, no rehearsal) is qualitative and the code was not retained: "converges faster on new classes", "forgets less", "embeddings move less in a 3D PCA". Everything below is a *re-derivation from scratch*. **This is the flagship gate, and it costs under a GPU-hour per full 25-run configuration** — so it is run exhaustively.

**Protocol.** CIFAR-10, 2-task class-incremental. Phase 1: 100 epochs on 5 classes (25,000 train / 5,000 test). Phase 2: extend head by 5 rows, 50 epochs on the 5 held-out classes only, **no rehearsal, no exemplars, no logit distillation** in the headline configuration. Evaluate over the full 10-way head (Class-IL) and, separately, within-task (Task-IL) — report both. This 5+5 split is **non-standard** (Mammoth's canonical `seq-cifar10` is 5 tasks × 2 classes); it is justified as the replication target and always supplemented with the canonical 5×2 protocol (T1.5). Use **5 random class partitions × 5 seeds = 25 runs per configuration**; at 2 tasks the partition choice moves the numbers more than the seed does, and a single "first five vs last five" split is indefensible.

**The stability–plasticity frontier is the primary object, not the endpoint (zero extra compute). Non-negotiable, and it applies at every tier including ImageNet.** Checkpoint and evaluate at **every phase-2 epoch** (at ImageNet scale: every epoch of every task, on a fixed 10,000-image validation subsample), plus every 1/4 epoch for the first 2 epochs of each new task, where the collapse actually happens — recording (new-class accuracy, old-class accuracy, Class-IL accuracy, Task-IL accuracy) at each. A full 10,000-image test pass through a 7.9M MLP is ~20 ms; 56 evaluation points per run cost under 2 seconds. At ImageNet/ResNet-18, 160 evaluations × 10k images ≈ 0.05 GPU-h per run, budgeted into the per-run costs in §1.7.4. **An endpoint-only comparison between two models that converge at different rates is invalid**: it cannot distinguish "better frontier" from "different point on the same frontier". The fix costs essentially nothing and the frontier trace is the headline figure. Report:
- the raw trace in the (new-class acc, old-class acc) plane, one polyline per (arm, partition/order, seed), with the median trace bolded;
- **A@matched-P**: old-class accuracy at the epoch where new-class accuracy first reaches 90% of its own converged value (per-run, so each arm is read at its own plasticity level);
- **frontier AUC**: area under the upper-left Pareto envelope of the trace, a single scalar for dominance testing;
- the **dominance fraction**: fraction of the new-class-accuracy grid on which one arm's old-class accuracy exceeds the other's.

**Architectures (parameter-matched, identical shapes).** Both are residual stacks, because nGPT's block update is defined on a residual stream and a plain feedforward MLP has no place to put α. Input 3072 (flattened 32×32×3, per-channel standardized) → input projection to width p = 512 → B = 3 residual blocks, each an expand-contract MLP 512 → 2048 → 512 with ReLU (SwiGLU as a secondary variant) → penultimate feature z ∈ R^512 → head W ∈ R^{C×512}. ≈ 7.9M parameters. Width p ∈ {256, 512, 1024} × depth B ∈ {2, 3, 6} as the Tier-1 grid, since the drift measurement's feasibility depends directly on p (§1.2). **VRAM: < 2 GB even at p = 1024, B = 6.** The entire CIFAR-10 train+test set is held resident on the GPU as a single fp16 tensor (50,000 × 3072 = 300 MB), so there is no dataloader in the loop and 12–16 runs execute concurrently on one A6000 (6–8 on a 3090).

- **MLP (baseline).** Blocks are `h ← h + f(h)`, RMSNorm before each block (a no-norm baseline diverges and would be a straw man; report both `MLP-RMSNorm` and `MLP-none`). Linear head with bias, unnormalized weights, softmax cross-entropy.
- **nMLP.** Faithful nGPT port, explicit about what is normalized:
  1. **Hidden states.** L2-normalized after *every* block: `h ← Norm(h + α ⊙ (Norm(f(h)) − h))`, with `α ∈ R^p` a **learnable per-coordinate eigen learning rate** (nGPT, 2410.01131), init 0.05, reported at convergence — nGPT observes α settling to 0.20–0.33; check whether CL training reproduces that.
  2. **Weights.** Unit-norm fan-in slices on every linear layer, enforced by retraction after every optimizer step, plus a learnable **per-row scale** `s` carrying the removed magnitude (nGPT's `s_u`, `s_v`), initialized so the layer's initial input–output scale matches the baseline's.
  3. **No LayerNorm/RMSNorm anywhere.** Redundant under (1); removal is part of the claim.
  4. **Classifier head.** Cosine: logit_c = s_z · ⟨ẑ, ŵ_c⟩, both unit-norm, `s_z` a learnable scalar logit scale (this scale *is* a vMF concentration; NormFace, 1704.06369, derives normalized softmax as a shared-κ mixture-of-vMF likelihood — cite it, do not re-derive it).
  5. **α ablation.** The eigen-LR mechanism is a residual-scaling change *independent of the sphere*; separately ablatable (§1.5 ablation 2-iv), reported both included and excluded.

**Head extension policy (a confound in its own right).** Row initialization must be **identical across methods**: three policies, all reported — (i) random unit vectors, (ii) small-Gaussian, (iii) **imprinting** with the new classes' mean normalized feature (Qi et al., CVPR 2018). Imprinting alone is known to reduce forgetting; if the nMLP advantage only appears under imprinting, the claim is Qi et al.'s. Cost: 45 phase-2-only runs = 0.6 GPU-h (T0.4). The winning policy is then fixed and used at every subsequent tier, and the paper states which.

**LR / effective-LR confound controls (mandatory at every tier, no exceptions).** Normalization changes the effective learning rate: for scale-invariant weights ∇L ⊥ w, ‖w‖ grows monotonically and **η_eff ≈ η/‖w‖²** decays for free (Arora, Li & Lyu, 1812.03981); weight decay on such weights is *only* an effective-LR knob (van Laarhoven 1706.05350; Hoffer 1803.01814; Li & Arora 1910.07454; Li–Lyu–Arora 2010.02916, only λη matters). **Every "normalized vs unnormalized" comparison is silently also a learning-rate comparison unless you sweep per arm.** At MLP scale a full per-arm sweep costs **4.6 GPU-hours**; at ResNet/ImageNet/PTM scale a shortened per-arm sweep costs 2.7 / 12.0 / 8.0 GPU-hours respectively (T2.1+T2.3, T3.1, T4.0). Those are all budgeted. **There is no arm anywhere in this plan that is compared at a shared LR**, and the paper should say so in exactly those terms, because it is the cheapest available differentiator from the existing literature.
- **(a) Per-arm optimum, never a shared LR.** MLP: grid η ∈ {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} × λ ∈ {0, 1e-4, 5e-4, 5e-3} = 24 configs, **swept independently for phase 1 and phase 2** (phase-2 sweep uses the coarser 12-config grid η × λ ∈ {0, 5e-4}). ResNet/ImageNet: η over 4–6 log-spaced values at a shortened 8-epoch-equivalent budget, λ over {5e-5, 5e-4} — the shortened-proxy assumption (that the LR ranking at 8 epochs matches the ranking at 16–32) is itself checked once, on 2 arms at full length, inside T3.5. Every arm is reported at *its own* best: phase-1 val accuracy for phase 1; **best phase-2 val accuracy on new classes only, selected without looking at old-class accuracy**, for phase 2. Selection that peeks at old-class accuracy is selection on the outcome and invalidates the comparison.
- **(b) Angular-update matching.** Log per-layer ‖Δw‖/‖w‖ per step as a **curve** (Kosson's rotational-equilibrium diagnostic, 2305.17212), both arms, both phases, at every tier. Then run a baseline whose λ is tuned so its angular-update curves overlay the nMLP's. This is the single control that separates "sphere" from "step size". Cost: 3 λ values × 5 seeds phase-1 + phase-2 ≈ 0.4 GPU-h at MLP scale, folded into T1.2; replicated once at ResNet-18/CIFAR-100 scale inside T2.4.
- **(c) Norm-matched control.** Baseline + WD tuned so ‖W‖ trajectories match nMLP's, without projection.
- **(d) Normalization-without-geometry.** Re-normalize weights each step but **block the tangent-space gradient projection** (project only in the retraction), isolating scale control from the spherical constraint.
- **(e) Riemannian control.** SGD-with-retraction vs plain SGD-then-renormalize; AdamW vs Adam-on-sphere. nGPT's own results use plain Adam + renormalize.
- **(f) Wall-clock honesty.** nGPT reports 60–80% higher per-step cost from unfused normalizations. Report **both** epochs-to-target and wall-clock-seconds-to-target at every tier. At ImageNet scale this matters materially to the budget itself, and the per-run costs in §1.7.4 already carry a 1.25× multiplier on normalized arms for it.

**Seeds, error bars, statistics.** MLP tier: 5 class partitions × 5 seeds = 25 runs/config. ResNet/CIFAR tier: 3 orders × 3 seeds. ImageNet tier: 2 orders × 3 seeds (§1.7.4 explains why 3 and not 5 — it is a budget decision and the paper states it). PTM tier: 5 seeds on Split-ImageNet-R, 3 on PTM Split-CIFAR-100. Report mean ± 95% bootstrap CI, **plus the per-run scatter**, never mean±std alone. Comparisons are **paired** on (partition/order, seed) and tested with a Wilcoxon signed-rank test on the paired differences; report the effect size (median paired difference and its CI), not just a p-value. Per-order variance is reported separately from per-seed variance (2509.22580, *The Lie of the Average*) — this matters most at ImageNet, where 2 orders × 3 seeds is thin and conflating the two variance sources would be the easiest way to overclaim. Frontier comparisons are paired on the same cells using frontier AUC and A@matched-P. **Power note:** at n = 6 cells (ImageNet), a paired Wilcoxon has ~0.8 power only for effects ≳ 1.5 σ_paired; the ImageNet tier is therefore positioned as *confirmation that the CIFAR/Tiny effect survives at scale*, not as the tier that establishes the effect. The establishing evidence is Tier 0–2, where n = 25 and n = 9.

**Deliverable of 1.1.** A four-panel result: (1) the (new-class acc, old-class acc) frontier traces, 25 runs per arm; (2) final Class-IL and Task-IL accuracy on old and new classes; (3) steps- **and seconds**-to-90%-of-final phase-2 accuracy (plasticity); (4) forgetting F_2 and BWT (stability) — each at every arm's own tuned (η, λ).

---

### 1.2 Quantifying the representation-stability finding — inference-only, every tier

This is the methodological core and the reason this program lives in a vMF repo. The existing evidence is a 3D PCA projection; 3 of 512 dimensions is not a measurement. **Total cost across all five tiers: ~15.5 GPU-h**, because everything here is inference on cached features.

**Setup.** Cache normalized penultimate features on a fixed, order-stable evaluation set of **old-class** inputs at every task boundary and at every intra-task checkpoint, so the drift measurement has a *trace* matching the accuracy frontier. Primary set: old-class *train* split (n matters — see the feasibility table); old-class *val/test* split as a generalization check. At ImageNet scale, subsample the old-class train pool to n = 200,000 (still n/p = 391 at p = 512, n/p = 98 at p = 2048 — both comfortably inside the regime) to keep feature caching and Procrustes tractable; state the subsample size. For unnormalized baselines, L2-normalize post-hoc so the comparison is well posed, and report the discarded magnitude separately as the per-sample log-norm ratio log‖z⁽²⁾‖ − log‖z⁽¹⁾‖ (itself a forgetting signal).

**Statistics computed (all inference-only, all on cached features).**

1. **Angular displacement.** cos θ_i = ⟨ẑ_i⁽¹⁾, ẑ_i⁽²⁾⟩; full distribution and median displacement in degrees. This alone replaces the PCA eyeball.
2. **Tangent drift direction.** u_i = (ẑ_i⁽²⁾ − ⟨ẑ_i⁽²⁾, ẑ_i⁽¹⁾⟩ ẑ_i⁽¹⁾)/‖·‖ ∈ S^{p−1} ∩ (ẑ_i⁽¹⁾)^⊥. **Primary statistic: the mean resultant length** R̄ = ‖(1/n)Σ u_i‖ (repo: `mean_resultant_length`) — a sufficient statistic, estimator-free, exactly calibratable by simulation. **Secondary:** κ̂ from `sra_kappa` / `banerjee_kappa` (NumPy) or `sra_kappa_torch` (Torch), reported *only* in the regime below.
3. **Rigid-vs-residual decomposition** (the operationalization of "the MLP twists the entire space"). Fit orthogonal Procrustes R̂ = argmin_{R ∈ O(p)} ‖Z⁽²⁾ − Z⁽¹⁾R‖_F, report the disparity (Ding et al., NeurIPS 2021, rotation-invariant only — exactly right here), then recompute (1) and (2) on the *residual*. Large raw drift with small residual = the space rotated rigidly (labels recoverable, old head merely misaligned — the Davari story). Large residual = the representation was destroyed. **Prediction: MLP has large drift and large residual, nMLP has small drift. If the MLP's drift is nearly rigid, the honest headline changes from "nMLP preserves features" to "nMLP preserves head–feature alignment" — LUCIR's story — and must be reported as such.** (Cost note: at p = 2048 the SVD for Procrustes is on a 2048×2048 Gram-side matrix, trivial; the expensive part is the n×p feature cache, which is why n is subsampled.)
4. **Class-conditional concentration as a continuous NC1 metric.** Per old class c, fit vMF and report κ̂_c at successive checkpoints — a dimension-aware version of neural-collapse NC1 (Papyan et al., PNAS 2020). Loosening of κ̂_c between phases is a clean forgetting signature. **But see the feasibility table: per-class n is the binding constraint almost everywhere, and at n ≤ p this statistic is dominated by the null.** ImageNet-1k with ResNet-18 (1,300 train images/class at p = 512) is the **only** setting in the entire program where per-class κ̂_c is even arguably estimable — which is itself a reportable finding, because the field reports per-class concentration routinely on CIFAR-100 and Tiny-ImageNet where it cannot be estimated at all. Also report between-class prototype geometry (pairwise cosines vs simplex-ETF ideal), which has no such problem.
5. **Layerwise.** All of the above at every block output (Ramasesh et al., 2007.07400, show forgetting concentrates in deeper layers). If the nMLP's advantage is uniform across depth, that is a different and more interesting mechanism.
6. **Never a per-input κ̂.** n = 1 gives R̄ = 1 exactly and κ̂ is a division by zero. Per-input statistics are the raw cos θ_i, or a **vMF log-likelihood** under a (μ̂, κ̂) fitted on a population — never an estimated per-input concentration.

**The matched null — and the subtlety that makes it non-trivial.** The naive null "n uniform vectors on S^{p−1}" is the *baseline* requirement, and it is mandatory because the finite-sample bias in R̄ and κ̂ is systematically **UPWARD** — exactly the direction that flatters a "more concentrated than uniform" claim. But here it is also not quite sufficient: u_i is constrained to the tangent space at ẑ_i⁽¹⁾, and the ẑ_i⁽¹⁾ occupy a class-structured cone, so under isotropic drift the pooled {u_i} are *not* marginally uniform and their pooled R̄ has a non-zero, geometry-dependent mean. The correct null: for each i, sample g ~ N(0, I_p), set u_i^null = Norm((I − ẑ_i⁽¹⁾ẑ_i⁽¹⁾ᵀ)g) — **uniform in the observed tangent space at the observed anchor, same n, same estimator**, B = 1,000 replicates (batched on GPU; seconds with the repo's samplers, minutes at p = 2048 with n = 200k). Report R̄_obs against the null mean and 95% interval, the standardized effect (R̄_obs − E_null[R̄])/sd_null, and a Monte-Carlo p-value. Same for κ̂. A second null — random pairing of θ₁ and θ₂ features across samples — controls for anchor structure. **No κ̂ and no R̄ is ever reported without its matched null next to it**; the repo benchmark established the upward bias directly.

**Feasibility regime — κ/p arithmetic for every architecture in this plan.** Accuracy is governed by **κ/p**, not p. Above κ/p ≈ 0.2, n = 10–25p buys ~1% relative accuracy; below κ/p ≈ 0.05, even n = 10⁵ does not, and the bias is upward. Penultimate widths in this plan: MLP 256/512/1024, ResNet-18 512, **ResNet-50 2048**, ViT-S 384, **ViT-B 768**.

| Measurement | p | n available | n/p | κ for κ/p ≥ 0.2 | Estimable? |
|---|---|---|---|---|---|
| MLP p=256, CIFAR-10 5+5 drift, old-class **train** | 256 | 25,000 | 97.7 | ≳ 51 | **Yes** |
| MLP p=256, old-class **test** | 256 | 5,000 | 19.5 | ≳ 51 | Yes, inside the 10–25p band |
| MLP p=512, old-class train / test | 512 | 25,000 / 5,000 | 48.8 / 9.8 | ≳ 102 | Train yes; **test marginal** |
| MLP p=1024, old-class train / test | 1024 | 25,000 / 5,000 | 24.4 / **4.9** | ≳ 205 | Train marginal; **test NOT estimable — R̄ + null only.** Bessel order ν = p/2 − 1 = **511**, exactly the cliff where `scipy.special.ive` underflows to zero; `sra_kappa` with `scipy_log_iv` returns NaN and safeguarded Newton stalls at ~1e-3 relative error. **CUSF / pure-torch log-Bessel mandatory at p ≥ 1024** |
| ResNet-18, Split-CIFAR-10 5×2, old **train** (8 cls × 5,000) | 512 | 40,000 | 78.1 | ≳ 102 | Yes |
| ResNet-18, Split-CIFAR-10 5×2, old **test** (8 cls × 1,000) | 512 | 8,000 | 15.6 | ≳ 102 | Yes, inside the band |
| ResNet-18, Split-CIFAR-100 10×10, old **train** (90 × 500) | 512 | 45,000 | 87.9 | ≳ 102 | Yes |
| ResNet-18, Split-CIFAR-100, old **test** (90 × 100) | 512 | 9,000 | 17.6 | ≳ 102 | Yes, inside the band |
| ResNet-18, Split-Tiny-ImageNet 10×20, old **train** (180 × 500) | 512 | 90,000 | 175.8 | ≳ 102 | Yes |
| ResNet-18, Split-Tiny-ImageNet, old **test** (180 × 50) | 512 | 9,000 | 17.6 | ≳ 102 | Yes, inside the band |
| **ResNet-18, Split-ImageNet-1k 10×100, old train** (900 × ~1,300, subsampled) | 512 | 200,000 | **390.6** | ≳ 102 | **Yes — the best regime in the entire program** |
| **ResNet-18, Split-ImageNet-1k, old val** (900 × 50) | 512 | 45,000 | 87.9 | ≳ 102 | Yes |
| **ResNet-50, Split-ImageNet-1k 10×100, old train** (subsampled) | **2048** | 200,000 | 97.7 | **≳ 410** | Yes on n. **But ν = p/2 − 1 = 1023 — twice past the `ive` cliff. Every off-the-shelf κ pipeline returns NaN or silently wrong values here. CUSF / torch log-Bessel is not optional** |
| **ResNet-50, Split-ImageNet-1k, old val** (900 × 50) | 2048 | 45,000 | 22.0 | ≳ 410 | Yes, inside the band; same ν = 1023 numerics |
| ViT-S/16 CLS, Split-CIFAR-100 10×10, old train / test | 384 | 45,000 / 9,000 | 117.2 / 23.4 | ≳ 77 | **Yes, both** — the narrow CLS width is an *advantage* here |
| **ViT-B/16 CLS, Split-ImageNet-R 10×20, old train** (180 × ~120) | **768** | 21,600 | 28.1 | ≳ 154 | Yes, above the band |
| **ViT-B/16 CLS, Split-ImageNet-R, old test** (180 × ~30) | 768 | 5,400 | **7.0** | ≳ 154 | **Marginal / below the band — R̄ + null only, no κ̂** |
| **ViT-B/16 CLS, PTM Split-CIFAR-100 10×10, old train** | 768 | 45,000 | 58.6 | ≳ 154 | Yes |
| **Per-class κ̂_c, ResNet-18 ImageNet-1k** (train, 1,300/class) | 512 | 1,300 | **2.54** | ≳ 102 | **Marginal — the only per-class cell in the plan that is not hopeless.** Report with its null, alongside a pooled common-κ fit, and state the bias explicitly |
| **Per-class κ̂_c, ResNet-50 ImageNet-1k** (train, 1,300/class) | 2048 | 1,300 | **0.63** | ≳ 410 | **NO. n < p.** Width kills it even at ImageNet sample counts |
| Per-class κ̂_c, CIFAR-10 MLP p=512 (train, 2,500/class) | 512 | 2,500 | 4.9 | ≳ 102 | **Marginal.** Pool with a common-κ fit; report per-class scatter |
| Per-class κ̂_c, CIFAR-100 ResNet-18 (train, 500/class) | 512 | 500 | **0.98** | ≳ 102 | **NO. n < p.** R̄ has an O(√(p/n)) floor; the statistic is *dominated by the null* |
| Per-class κ̂_c, CIFAR-100 ResNet-18 (test, 100/class) | 512 | 100 | **0.20** | — | **NO, categorically. Do not compute it** |
| Per-class κ̂_c, Tiny-ImageNet (test, 50/class) | 512 | 50 | **0.10** | — | **NO, categorically** |
| Per-class κ̂_c, ViT-B/16 ImageNet-R (train, ~120/class) | 768 | 120 | **0.16** | — | **NO, categorically** |

**This table is a result, not just a planning artifact,** and the ImageNet rows sharpen it. The per-class rows show that the field routinely reports per-class concentration on exactly these benchmarks with exactly these sample counts, and at 100 test images per class in R^512 the reported number is a function of p/n, not of the representation. The ResNet-50 rows show that **going to a wider backbone makes the measurement worse, not better**, even though it makes the model better — 2048-dim features at 1,300 images/class are strictly less measurable than 512-dim features at the same n. And the ν = 1023 row shows that the standard SciPy path is silently broken for every ResNet-50/ImageNet concentration number anyone has ever published. Publishing this boundary — with the matched null demonstrating it — is a contribution in its own right.

**Fallback when κ/p < 0.05 or n ≲ p** (planned for, not discovered): do **not** report κ̂. Report (i) R̄ with its matched null and standardized effect, valid at any concentration and any n; (ii) κ̂ restricted to the top-k PCA subspace of the phase-1 features (k ∈ {32, 64, 128}), where n/k ≥ 10 even for the worst per-class case retained — **stating explicitly that this changes the estimand** from "drift coherence in R^p" to "drift coherence within the dominant feature subspace", and reporting the fraction of drift energy the subspace captures; (iii) the numerical-validity finding (SciPy's silent `ive` underflow at ν > 511) as a reportable methodological result, validated against an mpmath brute-force reference at ν ∈ {127, 255, 511, 1023}.

**Paired non-vMF metrics (mandatory — the claim never rests on κ̂).** Linear and RBF CKA (Kornblith et al., 1905.00414), with the caveat from *Reliability of CKA as a Similarity Measure* (ICLR 2023) that CKA is outlier- and high-variance-direction-dominated — never reported alone; orthogonal Procrustes disparity and Williams et al.'s generalized shape metrics (NeurIPS 2021) with their bootstrap tests; principal angles between top-k PCA subspaces with the Grassmann distance; PWCCA; and **linear-probe representation forgetting** (Davari, 2203.13381): re-fit a linear head on the post-sequence backbone for the old classes, report probe accuracy vs deployed accuracy. That gap *is* the head-vs-feature decomposition, and it is the single most likely way this paper's mechanism story gets reframed. At ImageNet scale the probe is a logistic regression on 200k cached 512-d (or 2048-d) features — minutes on GPU, budgeted in T3.7.

---

### 1.3 Scaling the architecture

The ladder is **MLP → ResNet-18 → ViT-S/16 retrofit → ResNet-50 (ImageNet) → pretrained ViT-B/16 (PTM tier)**. What remains cut: **ViT from scratch on ImageNet-1k** in any form. A DeiT-S 300-epoch recipe is ~300–500 A100-h per run; even a 100-epoch cut-down is ~120 A100-h ≈ 300 A6000-h for a *single* arm, which is three-quarters of this entire program for one data point. ViT appears here only as (a) CIFAR/Tiny-ImageNet from scratch at ViT-S scale, or (b) *pretrained* ViT-B/16 checkpoints in the PTM tier. That boundary is firm.

| Stage | "Normalized" means concretely | What breaks | Cost / CIL run (A6000) | Peak VRAM |
|---|---|---|---|---|
| **nMLP** (7.9M) | §1.1 | Nothing; α occasionally collapses to ~0 under long phase-2 training, itself a finding | **0.035 GPU-h** (5+5), 0.053 (5×2) | < 2 GB |
| **n-ResNet-18** (11.2M, from scratch, 32×32 CIFAR stem) | Unit-norm fan-in slice per conv filter with a learnable per-filter scale; **all BatchNorm removed**, replaced by magnitude-preserving ops and forced weight normalization following EDM2 (Karras et al., 2312.02696); residual blocks use `h ← Norm(h + α ⊙ (Norm(F(h)) − h))`; cosine head with learnable s_z | BN removal costs raw accuracy and needs LR re-tuning; BN's *running statistics* are themselves a documented CL forgetting vector, so removing BN gives the nMLP a free advantage — **the baseline set must include a BN-stats-frozen-after-task-1 ResNet-18 and a GroupNorm ResNet-18**, or the comparison is unfair. GAP features must be normalized before the head | **0.17** (S-CIFAR-10 5×2), **0.26** (S-CIFAR-100 10×10), **0.70** (S-Tiny-IN 10×20) | ~6 GB @ batch 256, AMP |
| **n-ResNet-18, ImageNet stem** (11.7M, 7×7 stem, 176²→224² progressive) | As above | Same, plus: BN-free ResNets at ImageNet scale are known to need careful LR/warmup (NF-ResNet lineage); budget the T3.1 sweep accordingly and **report the joint-training top-1 of every arm** so a reviewer can see the BN-removal cost separately from the CIL effect | **1.5** (10×100 B0, 16 ep/task-equiv, incl. per-epoch frontier eval) | ~14 GB @ batch 256, AMP, channels_last |
| **n-ResNet-50, ImageNet stem** (25.6M) | As above | Same; p = 2048 makes the §1.2 measurement *harder* (see table) — this arm is a confirmation of the accuracy result and a stress test of the log-Bessel path, not the primary measurement vehicle | **4.5** (10×100 B0, 16 ep/task-equiv) | ~22 GB @ batch 256, AMP — **fits an A6000 comfortably; on a 3090 requires batch 128 + grad-accum**, which changes BN-free optimization enough that the 3090 numbers must be re-tuned, not transferred |
| **n-ViT-S/16 retrofit** (22M, ImageNet-21k pretrained) | Row-normalize every linear as an *exact, function-preserving reparameterization* — write W = diag(‖w_i‖)·Ŵ and fold magnitudes into a learnable per-row scale — plus CLS-feature normalization, cosine head, α on both residual branches. **LayerNorm retained.** | This is **not** full nGPT and must never be described as such: LN cannot be folded. The exactness is verifiable at zero cost (assert max \|logit difference\| < 1e-4 at init) and should be *reported as a check*, because it makes the arm a clean isolation of "normalized parameterization + eigen-LR" with the pretrained function held fixed at step 0 | **0.30** (S-CIFAR-100 10×10, 224², batch 64, 10 ep/task) | ~10 GB @ batch 64, AMP |
| **n-ViT-B/16 retrofit** (86M, ImageNet-21k pretrained) | Same exact reparameterization at ViT-B width (p = 768); this is our arm in the PTM tier | The PTM baselines (L2P/DualPrompt/CODA) freeze the backbone and train prompts; our retrofit trains the backbone. **That is not a matched comparison and the paper must add a prompt-tuned retrofit variant** (normalized parameterization, backbone frozen, prompts + cosine head trained) so there is an arm that is protocol-matched to L2P/DualPrompt. Both variants are in the 8-arm Tier-4 set | **1.2** (S-ImageNet-R 10×20, 224², batch 64) | ~18 GB @ batch 64, AMP (frozen-backbone arms ~11 GB) |
| ~~n-ViT-full (LN removed)~~ | — | Requires re-adaptation of the destroyed pretrained checkpoint | **DEFERRED** — §1.9 D1 | — |
| ~~ViT/DeiT from scratch on ImageNet-1k~~ | — | — | **OUT** — §1.9 D2 | — |

The MLP → ResNet-18 step is the one that has to work. The ViT-S retrofit shows the mechanism survives (a) a transformer, (b) attention, and (c) a *pretrained* initialization at 0.30 GPU-h per run. The ImageNet-1k tier is the scale credential. The PTM tier is the SOTA credential.

---

### 1.4 Scaling the benchmark

Benchmark ladder: **Split-CIFAR-10 → Split-CIFAR-100 → Split-Tiny-ImageNet → Split-ImageNet-1k → Split-ImageNet-R (+ PTM Split-CIFAR-100)**. All at **≥ 2 task orders**, with per-order variance reported separately from per-seed variance (2509.22580).

- **Split-CIFAR-10 5+5** (the replication) and **canonical 5 tasks × 2 classes** (Mammoth `seq-cifar10`), Class-IL and Task-IL — MLP tier at 3 orders × 5 seeds (T1.5), ResNet-18 tier at 3 orders × 3 seeds (T2.2).
- **Split-CIFAR-100 10×10, from-scratch (B0)**, 3 class orders × 3 seeds, ResNet-18 (T2.4) and ViT-S/16 retrofit (T2.7). **Only the 10×10 B0 protocol** at this tier; 20×5 and 5×20 remain cut. The paper states which convention every number uses and does not claim comparability with B50 tables.
- **Split-Tiny-ImageNet 10 tasks × 20 classes**, 64×64, ResNet-18 from scratch, 2 orders × 3 seeds, 4 methods (T2.5). This is the long-sequence stress test **and the Gate-2→3 predictor for ImageNet**, which is why its seed count was raised from 2 to 3 relative to the previous version.
- **Split-ImageNet-1k, ResNet-18 primary and ResNet-50 confirmation** (Tier 3). Two protocols, because the two halves of the literature use different ones and a reviewer will want both:
  - **10 tasks × 100 classes, B0** (the iCaRL/DER/FOSTER convention), 2 orders × 3 seeds, 6 methods, ResNet-18 (T3.2); 1 order × 3 seeds, 4 methods, ResNet-50 (T3.4).
  - **B500 + 5 × 100** (the LUCIR/PODNet large-base convention), 1 order × 3 seeds, 4 methods, ResNet-18 (T3.3). Large-base is the regime where cosine heads were originally shown to matter, so it is the fairest possible ground for the LUCIR-lite baseline — include it precisely because it is unfavorable to us.
  - **Joint oracles** trained at the identical budget (T3.6) so intransigence I_k is reportable and so the reader can see the short-schedule ceiling explicitly.
  - **Epoch-budget control (T3.5):** the whole ImageNet tier runs at 16 epochs/task-equivalent. Re-run 3 methods × 3 seeds at 32 epochs/task-equivalent on one order to verify the arm *ranking* is budget-invariant. **If the ranking flips between 16 and 32 epochs, the short-schedule ImageNet result is not reportable** and must be dropped or re-run at 32 throughout (which would cost +54 GPU-h and is covered by the ceiling headroom). This control is what makes the short-schedule choice defensible rather than convenient.
- **Split-ImageNet-R 10 × 20** (~24k train / ~6k test images, 200 classes), pretrained ViT-B/16, 1 order × 5 seeds, 8 arms (T4.1), plus **PTM Split-CIFAR-100 10×10**, 1 order × 3 seeds, same 8 arms (T4.2). ImageNet-R is the benchmark the PTM-CIL literature has standardized on and it is *small* — 30k images total — which is exactly why cutting this tier for cost was the wrong call.
- ~~CORe50~~, ~~OmniBenchmark~~, ~~Split-ImageNet-A~~, ~~VTAB~~: still cut (§1.9 D5).

Every benchmark carries the §1.2 drift measurement at the (n, p) regime named in the feasibility table, computed at **every task boundary**, at near-zero marginal cost. That is the cheapest strength this project has.

---

### 1.5 Baselines and ablations

**Rehearsal-free baselines at MLP scale (T1.6, 8 methods × 5 partitions × 3 seeds = 5.3 GPU-h).** Lower bound: sequential finetuning. Upper bound: joint training (also supplies a_k* for intransigence). Regularization: **EWC** (1612.00796), **online EWC**, **SI** (1703.04200), **MAS** (1711.09601). Distillation: **LwF.MC** (1606.09282). Exemplar-free CIL: **FeTrIL** (WACV 2023) and **FeCAM** (NeurIPS 2023) — feature-space post-hoc methods, essentially free on cached features. Neural-collapse CL — the nearest competitors, because a fixed simplex-ETF classifier *is* a hyperspherical normalized classifier: **NC-FSCIL** (2302.03004) reduced to its frozen-ETF-head component (ablation 5). Directional-statistics CL: **Michel et al. (AAAI 2024)** vMF-mixture, cited and, at MLP scale, reimplemented.

**Baselines at ResNet-18 / CIFAR-Tiny scale (T2.4, 6 methods).** nResNet-18, BN baseline, **BN-stats-frozen-after-task-1**, GroupNorm baseline, cosine-head-only (LUCIR-lite), LwF.MC. Rehearsal methods (ER, iCaRL, DER++, GDumb) are run **only at MLP scale**, clearly labeled as an upper reference.

**Baselines at ImageNet-1k scale (T3.2, 6 methods).** Sequential finetune (lower bound), BN baseline, BN-stats-frozen, **cosine-head-only (LUCIR-lite)**, **LwF.MC**, n-ResNet-18. Plus the joint oracle from T3.6. The ResNet-50 confirmation (T3.4) and the B500 protocol (T3.3) each use the 4-method core: finetune, BN baseline, LUCIR-lite, n-ResNet. **Deliberately no rehearsal methods at ImageNet scale** — the claim is rehearsal-free, and adding iCaRL/DER at ImageNet would double the tier for a comparison the paper is not making.

**PTM baselines (T4.1/T4.2) — the trim, and why these five.** The previous plan's 9 baselines × 2 backbones × 3 benchmarks × 5 seeds was breadth, not per-run cost: a single L2P/DualPrompt/CODA-Prompt run on Split-ImageNet-R is ~1–1.5 GPU-h. Keep **one backbone (ViT-B/16, ImageNet-21k pretrained, the field's default), two benchmarks, 5 seeds on the primary**, and these arms:

| Arm | Why it is non-negotiable | GPU-h/run |
|---|---|---|
| **SimpleCIL** (frozen features + class-mean prototypes, from APER/2303.07338) | The "is any of this necessary?" control. Nearly free, and it embarrasses about half the literature. Omitting it is the single most common reviewer complaint in PTM-CIL | 0.2 |
| **L2P** (CVPR 2022) | The paper that created the subfield. A reviewer will name it | 1.0 |
| **DualPrompt** (ECCV 2022) | The G-prompt/E-prompt decomposition; the standard second row of every table | 1.1 |
| **CODA-Prompt** (CVPR 2023) | The strongest of the prompt family and the one current papers are actually measured against | 1.3 |
| **RanPAC** (NeurIPS 2023) | Near-training-free random-projection + class-prototype method that beats the prompt family on several of these benchmarks. **Directly adversarial to our thesis** — it is a frozen-feature method, so if it wins, "the backbone should not move" beats "the backbone should move on a sphere". Include it precisely for that reason, and it costs almost nothing | 0.4 |
| **EASE** (CVPR 2024) | The recent SOTA anchor; without a 2024-era method the table looks stale | 1.5 |
| **n-ViT-B/16 retrofit, full-backbone** (ours) | The architecture claim at PTM scale | 1.8 |
| **n-ViT-B/16 retrofit, frozen-backbone + prompts** (ours) | Protocol-matched to L2P/DualPrompt/CODA; without it the comparison is unmatched and a reviewer will say so | 1.0 |

**Dropped and why:** **SLCA** (full-backbone finetune with separated LRs — expensive per run, ~2.5 GPU-h, and largely subsumed by our full-backbone arm which occupies the same protocol slot); **APER/ADAM** (its strongest configuration *is* SimpleCIL, which we keep); **MOS** and **CL-LoRA** (2024–25 methods that a reviewer is less likely to demand than EASE, and each adds ~50 GPU-h across seeds and benchmarks for a third recent-SOTA data point). Each is listed in §1.9 D5 with its unlock cost, so the paper can add one in rebuttal for ~25 GPU-h if a reviewer insists.

**Ablations that would kill the claim** (1 and 2 are Tier 0/Tier 1 and gate everything downstream):

1. **Is it just an effective-LR effect?** The full (η, λ) grid of §1.1(a) plus angular-update matching §1.1(b). **Kill condition: the tuned baseline closes the gap to ≤ 1.5 points.** This is the deflationary null of the whole program (EDM2/Kosson framing) and the most likely outcome. It costs 4.6 GPU-h to test. Run it first.
2. **Is it just a cosine-classifier effect — i.e. LUCIR rediscovered?** Nine-way component decomposition (T1.1/T1.2, 12.7 GPU-h), each variant at *its own* tuned LR:
   (i) feature normalization only; (ii) weight normalization only; (iii) **cosine classifier only = LUCIR-lite**, plain backbone; (iv) eigen-LR residual scaling only, no normalization; (v) normalize-and-retract without tangent-space gradient projection; (vi) full nMLP; (vii) full nMLP with **per-row scales s ∈ R^{out}**; (viii) full nMLP with a **single global scalar per layer**; (ix) full nMLP with a frozen simplex-ETF head.
   **Kill condition: (iii) accounts for ≥ 80% of the gap with the residual CI including 0.** Given the literature, expect (iii) to be large. State up front how much of the effect is (iii); the contribution is whatever (i)+(ii)+(iv) add on top. LUCIR-lite is then carried as an arm at **every** subsequent tier including ImageNet, so the residual is measured at scale and not just at MLP scale.
   **(vii) vs (viii) is a cross-project result.** It is exactly the DeLoRA-vs-DoLoRA question — DeLoRA (arXiv:2503.18225, ICLR 2025, merged into HuggingFace PEFT) normalizes the low-rank update and learns a **single per-layer scalar boundary**, while the DoLoRA manuscript's remaining novel delta is **per-component scales s_i**, and DoLoRA's own Appendix A proves a global scale is a strictly worse approximator when the target's singular values are non-uniform. That question is architecture-agnostic and **does not require an 8B model**: the (vii)/(viii) contrast answers "does per-component strength beat a single global scale?" at **1.4 GPU-hours** on CIFAR-10 MLPs with 25 paired runs per arm. Cross-reference `03-dolora.md`; report the singular-value spectrum of the learned per-layer update alongside the accuracy delta, since the theory predicts the gap grows with spectral non-uniformity.
3. **Head vs. features** (Davari): freeze the phase-1 backbone and train only the new head; and linear-probe old classes from the post-sequence backbone. If normalized and unnormalized backbones probe equally well, the effect is entirely classifier bias — this reframes the paper as the measurement paper rather than killing it. Inference-only on cached features; ~0.2 GPU-h at MLP scale, folded into T2.8/T3.7/T4.3 at the larger tiers. **Run this at ImageNet scale specifically** — it is the ablation most likely to change the story, and ImageNet is where a reviewer will trust it.
4. **Norm-matched control** (§1.1c) and **Riemannian-optimizer control** (§1.1e). Folded into T1.1/T1.2.
5. **Fixed vs learned sphere structure:** variant (ix) — separates "hypersphere" from "pre-allocated prototype geometry" (NC-FSCIL's contribution).
6. **Parameter/capacity parity and identical init scale**; α-init sweep {0.01, 0.05, 0.2, 0.5} and α-frozen-at-init (4 settings × 5 seeds = 0.7 GPU-h, folded into T1.2).
7. **μP / νGPT parameterization.** Shigida, Hanin & Gromov show nGPT does **not** exhibit LR transfer across width or horizon. Since we sweep p ∈ {256, 512, 1024} and then jump to 2048 (ResNet-50) and 768 (ViT-B), we **re-tune per width at every tier** (T1.3, T2.1/T2.3, T3.1, T4.0) rather than transferring — and report that we did, because "we tuned at p = 512 and reused" is exactly the failure they document. This is also why the ResNet-50 arm gets its own share of T3.1 rather than inheriting ResNet-18's LR.

---

### 1.6 Metrics (exact definitions)

Let a_{i,j} = accuracy on task j after finishing task i; T tasks.

- **Final average accuracy** A_T = (1/T) Σ_j a_{T,j}.
- **Average incremental accuracy** (iCaRL) Ā = (1/T) Σ_i A_i — the curve average, systematically higher than A_T. State which one every number is; do not conflate. (ImageNet CIL tables in the literature are split between the two; report both, always.)
- **BWT** (GEM, 1706.08840) = (1/(T−1)) Σ_{j<T} (a_{T,j} − a_{j,j}). Negative = forgetting.
- **Forgetting measure** (RWalk, 1801.10112): f_j^k = max_{l<k} a_{l,j} − a_{k,j}; F_k = (1/(k−1)) Σ_j f_j^k. Uses the max over history, so F_k ≠ −BWT in general — report both and say so.
- **Intransigence** (RWalk): I_k = a_k* − a_{k,k}, with a_k* from a jointly-trained oracle **at the identical epoch budget**. We train the oracle at every tier including ImageNet (T3.6), so we report I_k; most papers omit it, which makes it a free differentiator — and at a short schedule it is the honest way to show how much of the gap to the ceiling is the schedule rather than the method.
- **Plasticity** = steps- **and wall-clock-seconds**-to-reach 90% of the converged new-task accuracy. FWT is near-degenerate in CIL and reported for Task-IL only.
- **Stability–plasticity frontier** (primary, §1.1): the (new-class acc, old-class acc) trace over all intra-task checkpoints; **A@matched-P**; **frontier AUC**; **dominance fraction**. Reported as a curve and three scalars, at every tier. **No invented scalar tradeoff score**, and **no endpoint-only comparison** — the endpoint is reported but never used alone to support a "better tradeoff" claim.
- **Drift metrics** (§1.2), on old-class inputs at every checkpoint: median angular displacement (degrees); R̄ of tangent drift directions with matched-null mean, 95% null interval, standardized effect z, and Monte-Carlo p-value; κ̂ (Sra two-Newton-step, CUSF/torch log-Bessel) **with its estimability verdict from the κ/p table**, or an explicit "not estimable at this (n, p)" cell; Procrustes disparity and post-Procrustes residual drift; linear and RBF CKA; top-k principal angles and Grassmann distance; pooled and per-class κ̂_c only where the feasibility table permits it; linear-probe old-class accuracy (Davari).
- **Efficiency:** GPU-hours per run **on an A6000** (and the 3090 multiplier), per-step wall-clock relative to the unnormalized baseline, peak VRAM, and **sustained images/sec for every ImageNet arm**. Every table in the paper carries the GPU-hour cost of the row. For the ImageNet tier, also report the joint-training top-1 of every backbone/arm so the short-schedule ceiling is visible.

---

### 1.7 Compute budget and timeline

#### 1.7.1 Hardware and accounting conventions

**Primary: one RTX A6000 (48 GB, 768 GB/s, ~155 TFLOPS bf16 tensor).** All numbers below are **serial A6000-hours** — the conservative accounting. MLP runs hold the whole dataset resident in VRAM and execute 12–16 concurrently, so Tiers 0+1 (40 serial hours) compress to ~4 hours of wall-clock. Nothing else is run concurrently.

Conversions used: 1 A100-h ≈ 2.5 A6000-h for convnet training with AMP + channels_last; 1 A6000-h ≈ 1.22 3090-h. All drift metrics are inference-only on cached features.

#### 1.7.2 Tier 0 — go/no-go, 10.0 GPU-h

| Item | Runs | GPU-h/run | Total |
|---|---|---|---|
| T0.1 Phase-1 (η, λ) grid: 24 cfg × 3 arms × 2 seeds | 144 | 0.022 | 3.2 |
| T0.2 Phase-2 (η, λ) grid: 12 cfg × 3 arms × 3 seeds (from cached θ₁) | 108 | 0.013 | 1.4 |
| T0.3 Headline 5+5: 3 arms × 5 partitions × 5 seeds, per-arm tuned | 75 | 0.035 | 2.6 |
| T0.4 Head-init policy: 3 policies × 3 arms × 5 seeds (phase-2 only) | 45 | 0.013 | 0.6 |
| T0.5 Frontier assembly + full §1.2 measurement + 1,000-replicate nulls + Procrustes/CKA/probes | — | — | 1.5 |
| Slack / reruns | — | — | 0.7 |
| **Tier 0 subtotal** | **372** | | **10.0** |

**Tier 0 decides the program.** It is 2.5% of the budget and it is the only tier that runs unconditionally.

#### 1.7.3 Tiers 1 and 2 — MLP decomposition and the CIFAR/Tiny ladder, 87.0 GPU-h

| Item | Runs | GPU-h/run | Total |
|---|---|---|---|
| T1.1 Decomposition LR sweep: 9 variants × 12 cfg × 2 seeds | 216 | 0.022 | 4.8 |
| T1.2 Decomposition headline: 9 variants × 5 partitions × 5 seeds (incl. controls 1.1b–e, α-init sweep) | 225 | 0.035 | 7.9 |
| T1.3 Width × depth LR **re-tune** (μP finding): 9 shapes × 2 arms × 4 η | 72 | 0.035 | 2.5 |
| T1.4 Width × depth headline: 9 shapes × 2 arms × 3 seeds | 54 | 0.062 | 3.3 |
| T1.5 Canonical Split-CIFAR-10 5×2, MLP: 3 arms × 3 orders × 5 seeds | 45 | 0.053 | 2.4 |
| T1.6 Rehearsal-free baselines, MLP: 8 methods × 5 partitions × 3 seeds | 120 | 0.044 | 5.3 |
| T1.7 §1.2 measurement over all Tier-1 checkpoints (incl. co-variation test across 18 cells) | — | — | 1.5 |
| Slack / reruns | — | — | 2.3 |
| **Tier 1 subtotal** | **732** | | **30.0** |
| T2.1 ResNet-18 LR sweep, Split-CIFAR-10: 5 arms × 6 η × 1 seed (shortened) | 30 | 0.09 | 2.7 |
| T2.2 ResNet-18 Split-CIFAR-10 5×2: 5 arms × 3 orders × 3 seeds | 45 | 0.17 | 7.7 |
| T2.3 ResNet-18 LR sweep, Split-CIFAR-100: 5 arms × 4 η × 1 seed (shortened) | 20 | 0.13 | 2.6 |
| T2.4 ResNet-18 Split-CIFAR-100 10×10 B0: 6 methods × 3 orders × 3 seeds | 54 | 0.26 | 14.0 |
| T2.5 ResNet-18 Split-Tiny-ImageNet 10×20: 4 methods × 2 orders × 3 seeds | 24 | 0.70 | 16.8 |
| T2.6 ViT-S/16 retrofit LR sweep, Split-CIFAR-100: 3 methods × 4 η | 12 | 0.17 | 2.0 |
| T2.7 ViT-S/16 retrofit Split-CIFAR-100 10×10: 3 methods × 2 orders × 3 seeds | 18 | 0.30 | 5.4 |
| T2.8 §1.2 measurement across all Tier-2 checkpoints and task boundaries | — | — | 2.0 |
| Slack / reruns (BN-removal re-tuning is the likeliest overrun) | — | — | 3.8 |
| **Tier 2 subtotal** | **203** | | **57.0** |

**3090 note.** Everything in Tiers 0–2 fits in 24 GB unchanged; expect ~1.22× the hours (≈ 106 3090-h for Tiers 0–2). The ViT-S retrofit at batch 64 / 224² peaks near 10 GB and is fine.

#### 1.7.4 Tier 3 — Split-ImageNet-1k from scratch, 214.0 GPU-h

**The cost model, stated so it can be checked.** With FFCV on local NVMe, one A6000 sustains roughly **4,800 img/s** for ResNet-18 training at 176² with AMP + channels_last + progressive resizing, and **~1,600 img/s** for ResNet-50 — i.e. **0.075 h and 0.22 h per ImageNet epoch** respectively. Reference point: FFCV trains ResNet-50 to ~75% top-1 in ~20 min on an 8×A100 node ≈ 2.7 A100-h ≈ 6–8 A6000-h; ResNet-18 has ~3× the throughput, giving 2–3 A6000-h for a comparable-length run. FFCV's published ladder for ResNet-18 is roughly **67% top-1 at 16 epochs** and **~71% at 40 epochs**; **we run at 16 epochs/task-equivalent and accept ~67%**, with T3.5 verifying that arm ranking is invariant to that choice.

**Why a 10-task CIL run costs the same as a 16-epoch joint run.** Each task trains on 1/10 of ImageNet. Ten tasks × 16 epochs × 128k images = 20.5M images = **16 full-ImageNet-epoch-equivalents**. So a whole 10×100 sequence costs what one 16-epoch joint run costs: 16 × 0.075 = 1.2 h for ResNet-18, plus ~0.05 h of per-epoch frontier evaluation on a 10k val subsample and ~0.05 h of feature caching at task boundaries, plus the 1.25× normalization overhead on normalized arms → **1.5 h/run** as a blended figure. ResNet-50: 16 × 0.22 = 3.5 h + overheads → **4.5 h/run**. This epoch-equivalence is the entire reason ImageNet CIL is affordable and it should be stated in the paper.

**Hard infrastructure prerequisite — check it before spending anything (Gate 2→3a).**
- **Storage.** The FFCV `.beton` for ImageNet-1k is ~**150 GB** at 256px/quality-90 and ~**350 GB** at 512px max side. It must live on **local NVMe**, not NFS, not a network volume, not a spinning disk. Budget ~400 GB free plus ~150 GB for the original JPEG tree during conversion.
- **Preprocessing.** Conversion is a one-time CPU job: ~1–2 hours on a 16-core machine. Not counted in GPU-hours but it is real wall-clock.
- **The check.** Run one full epoch of the ResNet-18 recipe and record sustained img/s and GPU utilization. **Require ≥ 3,500 img/s and ≥ 85% GPU util.** At 2,000–3,500 img/s, re-cost the tier upward before proceeding. **Below 2,000 img/s you are I/O bound, every number in this subsection is wrong, and Tier 3 must not be started** — the fallback is the 186-GPU-h Tier 0/1/2/4 program at ICLR/TMLR. DALI or WebDataset shards are acceptable substitutes for FFCV; the throughput requirement is the same and the check is identical.
- **If the machine has fewer than ~16 dataloader-capable CPU cores**, ResNet-18 at 4,800 img/s will not be reachable even from NVMe with a JPEG pipeline; this is the case FFCV's raw/compressed hybrid storage mode exists for, and it is why the throughput check is run on the actual recipe rather than a synthetic benchmark.

| Item | Runs | GPU-h/run | Total |
|---|---|---|---|
| T3.0 FFCV build + **throughput audit (Gate 2→3a)**: 3 pipeline configs × 1 epoch, R18 and R50 | — | — | 4.0 |
| T3.1 ResNet-18/50 LR + WD sweep, 8-epoch-equivalent single-phase proxy: 4 arms × 5 η × 1 seed | 20 | 0.60 | 12.0 |
| T3.2 **ResNet-18 Split-ImageNet-1k 10×100 B0**: 6 methods × 2 orders × 3 seeds | 36 | 1.50 | 54.0 |
| T3.3 ResNet-18 Split-ImageNet-1k **B500 + 5×100** (LUCIR/PODNet convention): 4 methods × 1 order × 3 seeds | 12 | 1.50 | 18.0 |
| T3.4 **ResNet-50 confirmation, 10×100 B0**: 4 methods × 1 order × 3 seeds | 12 | 4.50 | 54.0 |
| T3.5 **Epoch-budget control**: 3 methods × 1 order × 3 seeds at 32 ep/task-equiv, ResNet-18 | 9 | 3.00 | 27.0 |
| T3.6 Joint oracles at matched budget: ResNet-18 × 2 seeds, ResNet-50 × 2 seeds | 4 | — | 12.0 |
| T3.7 §1.2 measurement at p = 512 and p = 2048 across all task boundaries; log-Bessel validation at ν = 1023; ImageNet linear probes | — | — | 6.0 |
| Slack / reruns (15% — BN-free ImageNet training is the likeliest overrun in the whole plan) | — | — | 27.0 |
| **Tier 3 subtotal** | **93** | | **214.0** |

**3090 note.** ResNet-18 is fine. **ResNet-50 at batch 256 with AMP peaks near 22 GB** — it fits an A6000 comfortably and a 3090 only barely, and with BN removed and magnitude-preserving ops the activation footprint is somewhat higher. On a 3090, run ResNet-50 at batch 128 with gradient accumulation to an effective 256 and **re-tune the LR** (do not transfer it); expect ~1.35× the listed hours for that arm. Tier 3 on a 3090 is ≈ **270 GPU-h**.

#### 1.7.5 Tier 4 — PTM tier (pretrained ViT-B/16), 89.0 GPU-h

| Item | Runs | GPU-h/run | Total |
|---|---|---|---|
| T4.0 ViT-B/16 LR sweep on Split-ImageNet-R (our 2 arms + 2 tunable baselines): 4 arms × 4 η × 1 seed | 16 | 0.50 | 8.0 |
| T4.1 **Split-ImageNet-R 10×20**: 8 arms × 1 order × 5 seeds | 40 | 1.20 | 48.0 |
| T4.2 **PTM Split-CIFAR-100 10×10**: 8 arms × 1 order × 3 seeds | 24 | 0.80 | 19.2 |
| T4.3 §1.2 measurement at p = 768 across task boundaries, both benchmarks, incl. the "PTM papers' concentration claims do not survive a matched null" study | — | — | 2.0 |
| Slack / reruns (15%) | — | — | 11.8 |
| **Tier 4 subtotal** | **80** | | **89.0** |

**3090 note.** Frozen-backbone arms (SimpleCIL, L2P, DualPrompt, CODA-Prompt, RanPAC, EASE, our prompt-tuned retrofit) peak at ~11 GB and are comfortable. Our **full-backbone ViT-B/16 retrofit** at batch 64 peaks near 18 GB — fits a 3090, but with no headroom; drop to batch 32 + accumulation if it OOMs, and re-tune. This is also the reason SLCA (full-backbone finetune with a second LR group) was dropped rather than merely deprioritized. Tier 4 on a 3090 ≈ **109 GPU-h**.

#### 1.7.6 Program total

| Tier | Runs | GPU-h (A6000) | GPU-h (3090) | Share |
|---|---|---|---|---|
| Tier 0 — CIFAR-10 MLP go/no-go | 372 | 10.0 | 12 | 2.5% |
| Tier 1 — MLP decomposition and breadth | 732 | 30.0 | 37 | 7.5% |
| Tier 2 — ResNet-18 / ViT-S ladder (CIFAR, Tiny-ImageNet) | 203 | 57.0 | 70 | 14.3% |
| Tier 3 — **Split-ImageNet-1k from scratch** | 93 | 214.0 | 270 | 53.5% |
| Tier 4 — **PTM tier (ViT-B/16, ImageNet-R)** | 80 | 89.0 | 109 | 22.3% |
| **TOTAL** | **1,480** | **400.0** | **≈ 498** | **100%** |

**≈ 400 A6000-hours (≈ 500 3090-hours) against a 1,000-hour ceiling.** On one A6000 running continuously that is **~17 days of GPU time**; with MLP concurrency in Tiers 0–1, ~16 days of calendar compute. **Expected spend is lower than 400**, because every tier above 0 is gated: a Gate-0 failure terminates at 10 GPU-h plus the 4 GPU-h Tier-M pivot; a Gate-2→3a (I/O) failure terminates the ImageNet tier at 4 GPU-h and lands the program at 186 GPU-h; a Gate-2→4 failure removes 89.

**The 600 hours of headroom are deliberate and allocated in priority order:** (1) re-running Tier 3 at 32 epochs/task-equivalent throughout if T3.5 shows the arm ranking is budget-sensitive (+54 h and the short-schedule caveat disappears); (2) a second task order for the ResNet-50 arm (+54 h); (3) reinstating one dropped PTM baseline in rebuttal (+25 h each); (4) 5 seeds instead of 3 on the ImageNet 10×100 primary (+36 h). Any one of these can be bought from the headroom without re-planning.

**What is *not* in the budget and why the budget is honest anyway:** the 1–2 CPU-hours of FFCV preprocessing, the CPU-side mpmath log-Bessel reference computation, and analysis/plotting. None of these touch the GPU.

**What was *added relative to a conventional plan*, because it is nearly free:** full per-arm phase-1 **and** phase-2 (η, λ) sweeps at every tier; per-width LR re-tuning across the width × depth grid and at every architecture change; 5 partitions × 5 seeds on every MLP headline number; **every-epoch checkpointing and the resulting stability–plasticity frontier at every tier including ImageNet**; the head-init-policy sweep; the epoch-budget invariance control; joint oracles at matched budget so intransigence is reportable; and the (vii)/(viii) per-component-vs-global-scale contrast that answers the DoLoRA/DeLoRA question at MLP cost.

#### 1.7.7 Timeline (~0.8 FTE, one A6000)

- **Week 1:** rebuild MLP/nMLP, GPU-resident data pipeline, run all of **Tier 0** (10 GPU-h ≈ 1 day wall-clock with concurrency). *Gate 0 evaluated end of week 1.* Detect the effective-LR artifact here, not in month four.
- **Weeks 2–3:** freeze the §1.2 measurement pipeline against the repo's estimators — tangent-space matched null, κ/p feasibility audit, CUSF/torch log-Bessel validated at ν ∈ {127, 255, 511, 1023} against mpmath. **Freeze metric definitions before seeing Tier-1 results.** Pre-register gate thresholds in the repo. **In parallel: start the FFCV conversion and run the Gate-2→3a throughput audit now** — it is CPU/IO work, it is independent of the science gates, and discovering an I/O wall in week 8 would be a disaster.
- **Weeks 4–5:** **Tier 1** (30 GPU-h). *Gate 1 evaluated end of week 5.*
- **Weeks 6–8:** **Tier 2** (57 GPU-h): ResNet-18 CIFAR ladder → Split-Tiny-ImageNet → ViT-S retrofit. *Gates 2→3b and 2→4 evaluated end of week 8.*
- **Weeks 9–14:** **Tier 3** (214 GPU-h): T3.1 sweep → T3.2 ResNet-18 10×100 → T3.5 epoch-budget control (run this **early**, right after T3.2, because a ranking flip invalidates the rest) → T3.3 B500 → T3.4 ResNet-50 → T3.7 measurement.
- **Weeks 11–15 (overlapping, since Tier 4 is gated independently and the arms are small):** **Tier 4** (89 GPU-h). Interleave with Tier 3 whenever the ImageNet pipeline is idle.
- **Weeks 16–18:** writing; release the measurement toolkit (tangent-space null, κ/p feasibility checker, numerically-safe log-Bessel at ν > 511) as a standalone package on top of the repo. The toolkit is the durable artifact and survives every negative result below.

---

### 1.8 Risks and what a negative result looks like

**Risk 1 (highest, ~50%): the advantage is an effective-LR artifact.** After the joint (η, λ) sweep and angular-update matching, the nMLP's forgetting advantage shrinks to 1–2 points. This is the EDM2/Kosson deflationary null and the literature's default expectation. *Detection: end of week 1, for 10 GPU-hours.* *Mitigation:* pivot to **Tier M**.

**Risk 1b (~30%): the arms share a frontier.** The traces overlap; nMLP simply sits at a more-stable/less-plastic point because normalization slows adaptation. The endpoint difference is then a *stopping-time* difference and the original claim evaporates. This is invisible to endpoint-only evaluation, which is exactly why the frontier is the primary object and why it costs nothing to check at every tier. *Mitigation:* report it as a finding — "hyperspherical normalization moves you along the stability–plasticity frontier rather than improving it" is a clean, useful, publishable negative, and the drift measurement then explains *why*.

**Risk 2 (high, ~40%): it is the cosine classifier.** Ablation 2-(iii) accounts for most of the gap → LUCIR rediscovered. *Mitigation:* report it immediately and honestly; the residual contribution of weight normalization + eigen-LRs on top of a cosine head becomes the empirical claim, measured with the paired statistics of §1.1 at every tier including ImageNet, not asserted. If the residual is zero, the architecture claim is dead and the measurement paper survives.

**Risk 3 (moderate): the drift is real but not estimable.** κ/p lands below 0.05 on the drift directions, so κ̂ is biased upward and uninformative — in the direction that flatters us. *Mitigation:* R̄-with-matched-null is the primary statistic precisely because it stays valid; subspace-restricted κ̂ is the interpretable secondary with its estimand change stated. "Here is the regime where the field's favorite spherical statistic cannot be computed, and here is what to report instead" is a contribution — and it now comes with three concrete numerical findings: SciPy's `ive` underflow at ν > 511 (so every off-the-shelf κ pipeline is silently NaN or wrong at p ≥ 1024, **including every ResNet-50 feature-concentration number at ν = 1023**), the n < p per-class collapse on the standard CIFAR-100 / Tiny-ImageNet / ImageNet-R splits, and the counterintuitive result that **wider backbones are less measurable at fixed data**.

**Risk 4 (moderate): it is a head-drift effect, not a feature-drift effect.** Linear probes on the post-sequence backbone recover old-class accuracy equally well for both architectures (Davari). *This reframes rather than kills:* the finding becomes "hyperspherical normalization preserves head–feature *alignment*, and here is the measurement that separates alignment from feature destruction" — genuinely useful, and it explains why cosine classifiers work. The "representations are more stable" phrasing must then be dropped everywhere.

**Risk 5 (moderate): it does not scale past CIFAR.** The effect vanishes for ResNet-18 on Tiny-ImageNet or under 10-task sequences. *Mitigation:* Gate 2→3b exists exactly here, and it fires for 97 GPU-h rather than 400. If the effect is CIFAR-only, that is a short honest report, not a paper.

**Risk 6 (new, moderate, ~30%): the ImageNet result is a short-schedule artifact.** At 16 epochs/task-equivalent every arm is underfit, and underfitting can suppress forgetting differences (or manufacture them, if one arm converges faster). *Detection: T3.5, the 32-epoch control, run immediately after T3.2 and before the ResNet-50 and B500 spends.* *Mitigation:* the headroom pays for a full 32-epoch re-run (+54 h). *If the ranking flips and the headroom is unavailable, the ImageNet tier is reported as "inconclusive at this budget" and the paper drops to Tier 0–2 + Tier 4* — which is still an ICLR-grade paper but not a CVPR one. This is now the single largest risk to the venue claim, which is why the control is early and non-optional.

**Risk 7 (new, structural, ~25%): the I/O wall.** Gate 2→3a fails; the machine cannot sustain 3,500 img/s. *Mitigation:* checked in week 2–3, before any science spend, for 4 GPU-h. Fallback is the 186-GPU-h program and the ICLR/TMLR target. The reason this is a listed risk rather than an assumption is that the entire ImageNet cost model rests on it.

**Risk 8: a frozen-feature method wins.** RanPAC or SimpleCIL beats every arm on Split-ImageNet-R, including ours. This is a live possibility and it is thesis-adverse: "do not move the backbone" would beat "move the backbone on a sphere". *Mitigation:* report it. It does not touch the from-scratch tiers (where there is no pretrained backbone to freeze), and the honest framing becomes "hyperspherical normalization is the right inductive bias when the backbone must train, and frozen features are the right answer when a strong PTM is available" — which is a genuinely useful boundary statement and is more than most PTM papers offer.

**Wall-clock risk (certain, must be disclosed):** normalization is unfused; expect materially higher per-step cost as in nGPT. Any "converges faster" claim reported only in epochs will be attacked. Report seconds. The 1.25× multiplier is already in the ImageNet per-run costs; if the real overhead is nGPT's 1.6–1.8×, Tier 3 grows by ~35 GPU-h, which the headroom absorbs.

**Tier M — the measurement-paper pivot (~4 GPU-h, reuses existing checkpoints).** If Gate 0 or Gate 1 fails: (i) extend the §1.2 measurement to all cached Tier-0/1 checkpoints across the width grid (0.5 GPU-h); (ii) run the null-calibration study on published CL feature extractors' released checkpoints, or on the baselines already trained here, to show that uncalibrated concentration claims do not survive a matched null (1.5 GPU-h); (iii) build and validate the κ/p feasibility checker and the numerically-safe log-Bessel path against an mpmath reference (2 GPU-h, mostly CPU). Deliverable: *"A null-calibrated directional-drift metric for representation stability in continual learning, and the regime where it can and cannot be computed."* ICLR/TMLR-grade methods paper, fully funded at 14 total GPU-hours. It is the program's floor.

**What would make this a strong CVPR paper:** the cosine-classifier ablation shows a real residual gain from architecture-wide normalization; the nMLP/nResNet **dominates the frontier** rather than sliding along it; the effect holds on Split-CIFAR-100, Split-Tiny-ImageNet **and Split-ImageNet-1k 10×100 across 2 orders with a budget-invariant ranking**; our ViT-B retrofit is competitive with CODA-Prompt/EASE on Split-ImageNet-R; and the drift measurement lands in an estimable regime with a large, null-calibrated separation that **co-varies** with the accuracy gap across the width × depth grid. **What would make it a good paper even if all of that fails:** the measurement methodology — tangent-space matched null, κ/p feasibility map across seven architectures, the SciPy underflow finding at ν = 511 and ν = 1023, the n < p per-class collapse, the wider-is-less-measurable result, and the negative finding that existing "our embeddings are more concentrated" claims are not calibrated.

---

### 1.9 Deferred: what still needs more than one 48 GB GPU

Nothing in this subsection is in the 400-GPU-hour budget. Each item states its cost, an **H100 ask** where the deferral is genuinely a hardware limit, and the **specific result that would justify unlocking it**.

**D1 — nViT-full (LayerNorm actually removed, q/k normalized, α on both residual branches).** Removing LN destroys the pretrained checkpoint and requires re-adaptation on ImageNet-1k before any CL protocol can start: **~50 A100-h ≈ 30 H100-h for ViT-S, ~3× that for ViT-B**. Doing both backbones with 2 seeds and a small LR sweep: **H100 ask ≈ 150 H100-hours** (≈ 375 A6000-h, which alone would blow past the ceiling on top of the existing 400). **Unlock condition:** the exact, function-preserving ViT-S/ViT-B *retrofits* (which are free) show a gap over the cosine-head baseline, establishing that the normalized parameterization matters before we pay to remove LayerNorm. If the retrofit shows nothing, removing LN will not save it. This is the largest remaining gap between what we run and what the title claims, and the paper must say so in the limitations section.

**D2 — From-scratch normalized-ViT/DeiT-S ImageNet-1k pretraining.** A DeiT-S 300-epoch recipe is ~300–500 A100-h per run ≈ 180–300 H100-h; two arms × 2 seeds plus an LR sweep is **≈ 1,500 H100-hours**. **Permanently out of scope** at any plausible unlock; listed so no reviewer thinks it was overlooked. The ViT-S CIFAR/Tiny-from-scratch arms and the pretrained ViT-B retrofit together cover the transformer axis at 1/100th the cost.

**D3 — Full-fidelity ImageNet-1k CIL for absolute-accuracy claims.** Re-running Tier 3 at 88 epochs/task-equivalent (ResNet-18 ~71% joint, ResNet-50 ~78%) is ~5.5× the current tier: **≈ 1,180 A6000-h ≈ 340 H100-hours**. **H100 ask ≈ 340 hours.** **Unlock condition:** an accepted paper plus a reviewer demanding absolute-SOTA-comparable ImageNet numbers, or a T3.5 result showing rank instability that the +54 h 32-epoch re-run does not resolve. Until then the ImageNet tables are explicitly budget-matched relative comparisons.

**D4 — Language: does the CL result feed back to nGPT's native domain?** nGPT is a language architecture and "does your CL finding say anything about the setting nGPT was designed for?" is a fair reviewer question. The honest small-scale version fits on the same hardware: SmolLM2-135M or Pythia-160m, from scratch or LoRA-adapted, over a 4-domain sequential stream (Pile subsets: PubMed → GitHub → ArXiv → StackExchange), nGPT-normalized vs baseline, measuring per-domain perplexity forgetting and the same §1.2 drift statistics on the final hidden state (p = 576 for SmolLM2-135M, p = 768 for Pythia-160m — both estimable at n = 25k held-out tokens). **Cost: ~30 A6000-h** for 2 arms × 3 seeds × 4 domains including a per-arm LR sweep. **Affordable but deliberately not in this project's 400 hours** — it is a second paper's worth of scope and the vision ladder must land first. **No H100 needed.** **Unlock condition:** Gates 0–2 all pass *and* the frontier-dominance result holds on ImageNet-1k. Anything at 0.5B+ parameters is out of scope on this hardware and is not planned.

**D5 — Breadth we chose not to buy.** Each is affordable individually and each buys breadth, not mechanism. Listed with costs so any one can be added in rebuttal from the 600 hours of headroom.
- **Dropped PTM baselines**: SLCA (~30 A6000-h across both Tier-4 benchmarks and seeds; also the arm most likely to OOM on a 3090), MOS (~25 h), CL-LoRA (~25 h), APER/ADAM beyond SimpleCIL (~15 h). **Unlock: a reviewer names one.** Budget ~25 h each from headroom.
- **CORe50 (NC and NI) and OmniBenchmark 30×10**: ~80 A100-h ≈ 200 A6000-h. **Unlock: a positive Tier-3/4 result plus an explicit reviewer request.**
- **Split-ImageNet-A and VTAB** (the harder PTM benchmarks): ~40 A6000-h. Same unlock.
- **Additional ImageNet task orders and seeds**: +36 h for 5 seeds on the 10×100 primary, +54 h for a second ResNet-50 order. **Unlock: any reviewer comment about statistical power at n = 6 cells** — which is a comment we should expect, and which the headroom is explicitly reserved for.
- **Split-CIFAR-100 protocol breadth** (20×5, 5×20, B50+10×5): ~45 A6000-h. Low priority; the ImageNet B500 protocol (T3.3) already covers the large-base convention at a scale reviewers care about more.


---

## Project 2 — "Does Per-Component Strength Beat a Global Scale? A Rank-Resolved, Null-Calibrated Study of Normalized Low-Rank Adapters, from ViT-S to Qwen3.5-9B"

**Venue:** ICLR / NeurIPS (the rank-resolved mechanism study is the headline; vision is the powered tier, language is the mechanism-reproduction tier). **Status:** from-scratch reimplementation — the student's code is gone.

> ### COMPUTE ENVELOPE — READ FIRST
> **Hardware: one NVIDIA A6000, 48 GB.** For *compute* an A6000 is essentially an RTX 3090 (same GA102 family; ~155 vs ~142 TFLOPS dense bf16, slightly lower memory bandwidth). **The upgrade is memory, not speed.** Every vision timing below is therefore carried over unchanged from the 3090 costing; what changed is which models fit.
> **Total: 571 GPU-h committed if every gate opens, 657 with a 15% rerun contingency.** Hard ceiling 1000; we are at 66% of it in the worst case and 0.8% in the case where the first gate stops us.
> **The decisive minimum is 100 GPU-h** (implementation + Tier 0 + Tier 1) and is a complete, publishable vision paper on its own. **243 GPU-h** adds the powered language rank curve at Qwen3.5-4B and turns it into a cross-modality mechanism claim. The remaining 328 h buys the 9B mechanism-reproduction tier and ViT-L scale, both gated.
> **VRAM is rarely binding — wall-clock is.** Peak VRAM anywhere in the plan is ~41 GB (vLLM eval of Qwen3.5-9B with a fat KV cache); peak *training* VRAM is ~23 GB (Qwen3.5-9B LoRA, bf16, batch 4 × 256 with gradient checkpointing). Qwen3.5-4B LoRA trains at ~12 GB. All fit.
> **What the model policy changes.** All three language tiers are now Qwen3.5 (Apache 2.0): **Qwen3.5-4B** for the powered rank curve, **Qwen3.5-9B** for the flagship tier. The 9B tier is consequently reframed from *replication of the DoLoRA paper's LLaMA-3-8B number* to *mechanism reproduction on a current model* (§2.8, §2.10) — the tradeoff is stated explicitly there rather than glossed.
> **Newly out of scope relative to the previous revision:** (i) any full fine-tuning in language — the smallest policy-compliant model is 4B, whose full-FT footprint is ~64 GB (weights + grads + fp32 Adam) and does **not** fit 48 GB; the full-FT language reference moves to §2.11. (ii) **Qwen3.5-27B dense LoRA is ~56 GB in bf16 and does not fit either**; the 35B MoE is comparable. Both are deferred to the 2×A100 node.
> **Still deferred (§2.11):** the full un-subsampled commonsense170k at 9B, 27B / 35B-MoE, a second 9B-class model, ViT-H, full-FT in language, and a seed count at 9B large enough to resolve a sub-point effect.
> **Critically: nothing in this plan quantizes a base weight.** 27B QLoRA would fit in 48 GB, but NF4 changes `||W0||` and its across-layer distribution, which is exactly the quantity DeLoRA folds into its update and which T2.1 nominates as a candidate mechanism. Quantizing would mean studying a different method. Everything here is bf16 base weights.

**Scope of the model policy in this section, stated once.** The Qwen3.5-only rule governs every model that is *trained or finetuned* in language, and all three language tiers comply. It has no vision counterpart — there is no Qwen3.5 image backbone — so the ViT tiers keep `timm`/HF ViT-S/B/L, DINOv2 and CLIP. That is also what the policy's diversity carve-out wants here: T2.1's entire purpose is heterogeneity of *pretraining objective* across backbones, which collapses to nothing if every backbone comes from one family. Flagged in the open rather than buried, so it can be overruled.

**Framing decision, made before any code is written.** The DoLoRA draft's core parameterization is DeLoRA (Bini et al., arXiv 2503.18225, ICLR 2025), already merged into HuggingFace PEFT as `peft.tuners.delora`. DeLoRA's Eq. 5 is per-rank-one normalization `(b_i a_i^T)/(||b_i|| ||a_i||) = B Ξ A`; that is `u_hat_i v_hat_i^T` under another name. The *only* genuine delta is per-component strength `S = diag(s_1..s_r)` versus DeLoRA's single learnable scalar `λ` per layer — and DoLoRA's own Appendix A proves a global scale is a strictly worse rank-`r` approximator whenever the target's singular values are non-uniform. So the sharp, architecture-agnostic, falsifiable question is:

> **Does per-component strength beat a single global scale, and if so, at which ranks, in which modalities, and by how much?**

The method is renamed **DeLoRA-PC** (per-component) throughout: "DoLoRA" collides phonetically with DoRA and DeLoRA and is unusable. `DoLoRA` is used below only when referring to the draft's reported numbers.

**What the 48 GB envelope changes about the argument.** The 3090 version of this plan had to argue that the flagship-scale language experiment was *unnecessary* — that a rank curve in vision was a better scientific object than one 8B-class LLM number. That argument was true and it stays in the paper, but it was also load-bearing in a way that made a reviewer's job easy ("you did not run the experiment the paper you are critiquing ran"). We can now run a same-scale version of it. The structure that results is better than either extreme:

- **Vision carries the statistical weight.** 19 independent VTAB tasks, 6 ranks, 3 seeds, an exact built-in null at `r=1`, and per-arm LR sweeps. This is where a sub-point effect can actually be resolved (MDE ≈ 0.58 VTAB-mean points, §2.12).
- **Language at 4B carries the cross-modality mechanism claim** with enough seeds to have a noise floor at all (5 seeds, 4 ranks including the `r=1` null).
- **Language at 9B carries mechanism reproduction at flagship scale.** Two seeds at 9B cannot resolve a sub-point difference and we do not pretend otherwise. What two seeds at 9B *can* do is test whether the effect behind the draft's headline — **+7.63** — survives on a current model, against an MDE of ~1.3 points, a six-fold margin. The 9B tier is there to confirm or refute a large claimed effect, not to discover a small one, and it is honest about no longer being a like-for-like replication (§2.8).

That division of labour is stated in the introduction, not buried in the limitations.

---

### 2.1 Reimplementation and verification — **0.1 GPU-h (CPU-bound)**

**Exact specification.** For a frozen base weight `W0 ∈ R^{d×k}` (PyTorch `nn.Linear` stores `W0` as `[out=d, in=k]`):

- Parameters: `U ∈ R^{d×r}` with columns `u_i`; `V ∈ R^{k×r}` with columns `v_i`; `λ ∈ R^r`.
- Normalized factors: `u_hat_i = u_i / max(||u_i||_2, ε)`, `v_hat_i = v_i / max(||v_i||_2, ε)`, `ε = 1e-6`.
- Scales: `s_i = softplus(λ_i) − log 2`, `λ_i` initialized to **0**, so `s_i = 0` and `ΔW = 0` exactly at init.
- Update: `ΔW = (α/r) · U_hat diag(s) V_hat^T = (α/r) Σ_i s_i u_hat_i v_hat_i^T`; `W' = W0 + ΔW`.
- Init: Kaiming uniform (`a = sqrt(5)`, matching PEFT's LoRA-A convention) on **both** `U` and `V`. LoRA's `B = 0` init is unavailable because `u_hat` would be undefined.

Two spec details the draft states loosely, which change behavior and must be pinned down:

1. **`s_i` is not non-negative.** `softplus(λ) − log 2 ∈ (−log 2, ∞)`, so scales can go negative down to `−0.693`. This is arguably a feature (sign flips let a component reverse direction without `u_i` traversing the sphere), but "learnable non-negative per-component scales" as written in the draft is wrong. Implement the shifted softplus as specified, and add a `--strict-nonneg` variant (`s_i = softplus(λ_i − c)`, `c = 6`) as an ablation (T2.3) so the sign freedom is testable rather than assumed.
2. **The gradient at init is nonzero.** `ds_i/dλ_i |_{λ=0} = σ(0) = 0.5`, and `dL/ds_i = (α/r)⟨∇_W L, u_hat_i v_hat_i^T⟩ ≠ 0` for random Kaiming `u,v`. LoRA at `B = 0` has `dL/dA = B^T ∇_W L = 0`, so LoRA's `A` is frozen for the first step. Early dynamics differ *by construction*; state it in the paper rather than let a reviewer discover it.

**Module design.** A single `DeLoRAPCLinear(nn.Module)` wrapping the frozen base layer. Forward, fused, no materialization of `ΔW`:

```
h = F.linear(x, W0, b)                      # frozen path
z = x @ v_hat                               # [.., r]      O(kr)
h = h + (α/r) * ((z * s) @ u_hat.T)         #              O(dr)
```

Normalization is `F.normalize(U, dim=0, eps=1e-6)` computed **in fp32 even under bf16 autocast** — column norms in `R^768` in bf16 lose ~3 decimal digits and the tangent-projection test fails spuriously. This matters more at 9B, where `d = 4096` and `d = 12288` (MLP): the fp32 normalization is not optional there. Overhead is `O(r(d+k))` per forward, versus DoRA's `O(dk)` column-norm of `W0+BA`. `get_delta_weight()` returns the dense `ΔW` for merge/unmerge; `merge()` is exact and reversible.

**PEFT integration.** Do not write a tuner from scratch — **fork `peft/tuners/delora`**. It already implements `DeloraConfig`, `DeloraModel(BaseTuner)`, `DeloraLayer(BaseTunerLayer)`, and the `_create_and_replace` / `merge` / `unmerge` / `get_delta_weight` surface. Changing `λ: scalar` to `λ: Parameter(torch.zeros(r))` plus a flag for DeLoRA's `||W0||` folding is a ~60-line diff. Three payoffs: (a) DeLoRA becomes an *exactly code-matched* baseline sharing the same data path, eliminating the implementation confound that ruins most PEFT head-to-heads; (b) the `||W0||` factor becomes a one-flag ablation rather than a confound; (c) it runs unmodified under `transformers` + `trl` for the LLM tiers and merges cleanly for vLLM evaluation, which is what makes §2.9/§2.10 affordable at all. Register as `peft_type="DELORA_PC"`; ship out-of-tree with `peft>=0.18` pinned, `~800 LOC` including tests.

Target-module resolution must handle three naming conventions: `timm` ViT (`attn.qkv` fused), `transformers` ViT (`query`/`key`/`value` split), and Qwen (`q_proj,k_proj,v_proj,up_proj,down_proj`). The fused `qkv` case needs a slicing wrapper so that "target q,v only" means the same thing across backbones. Qwen3.5-9B interleaves two block types, so resolution matches on projection names actually present in each block and the adapter placement is **recorded per block type** in the run config; the per-block-type placement table goes in the reproducibility appendix so the parameter counts in the main table are auditable.

**Unit tests** (`pytest`, CPU-only, fp64 wherever a tolerance is asserted; total runtime target < 90 s):

| Test | Assertion |
|---|---|
| `test_zero_at_init` | `get_delta_weight()` is **bitwise** zero for 20 random `(d,k,r,dtype)` combos; wrapped forward is bitwise identical to base `nn.Linear`. |
| `test_scale_grad_analytic` | With `L = ⟨G, W'⟩`, autograd `dL/ds_i` matches `(α/r)⟨G, u_hat_i v_hat_i^T⟩` and `dL/dλ_i` matches that times `σ(λ_i)`, rel. err < 1e-12 in fp64; cross-checked against central differences at `h=1e-6`. |
| `test_tangent_projection` | `∇_{u_i} L = (α/r)(s_i/||u_i||)(I − u_hat_i u_hat_i^T) G v_hat_i` to 1e-12, **and** `\|⟨u_hat_i, ∇_{u_i} L⟩\| < 1e-13·||∇_{u_i} L||`. This orthogonality is the load-bearing geometric claim of the project; it gets its own test. |
| `test_gradcheck_exact` | `torch.autograd.gradcheck` in fp64 passes — no `detach()` anywhere in the graph. |
| `test_dora_exact_vs_detached` | Implement DoRA twice (exact `||W0+BA||_c`; reference-style detached norm). Assert the two gradient fields are *not* equal; record cosine similarity and relative L2 vs `||BA||_F/||W0||_F`. Promoted to **Figure 2**: it quantifies how approximate the published DoRA gradient actually is, and costs nothing. |
| `test_expressivity_appendix_A` | Constructive proof of the draft's Appendix A: for random rank-`r` `M = Σ σ_i p_i q_i^T`, set `u_i = p_i`, `v_i = q_i`, `s_i = r σ_i/α`; assert `||ΔW − M||_F/||M||_F < 1e-12`. Separately assert the `s_i > −log 2` bound costs no expressivity (absorb sign into `u_i`), and that the *global-scale* variant's best rank-`r` approximation error is bounded below by the singular-value dispersion of `M` — the quantitative version of the claim this whole project tests. |
| `test_reduces_to_delora` | With `λ` tied across components and `||W0||` folding on, outputs and gradients match stock PEFT DeLoRA to 1e-12. Proves per-component strength is the *only* difference. |
| `test_reduces_to_delora_at_r1` | At `r=1`, tied and untied are the same parameterization. Bitwise-identical trajectories under a fixed seed. This is the **built-in null** for the whole rank curve, in both modalities. |
| `test_merge_roundtrip` / `test_dtype_device_parity` | merge→unmerge identity to dtype epsilon; fused forward equals explicit-`ΔW` forward within 3e-3 (bf16) / 1e-6 (fp32). |
| `test_merge_matches_vllm` | Merged `W0+ΔW` written to a checkpoint and reloaded by vLLM produces logits matching the unmerged `transformers` forward to 5e-3 on 64 held-out prompts. Run on **both** Qwen3.5 block types. Cheap, CPU+1 GPU-min, and it is the failure mode that silently corrupts every LLM eval number in §2.9/§2.10. |
| `test_golden_trajectory` | Fixed seed, 20 AdamW steps on a 64×32 toy problem; hash of the loss trajectory pinned. Catches silent refactor regressions. |

**Cost:** ~1 engineer-week, CPU only, **< 0.1 GPU-h**. Nothing here needs training.

---

### 2.2 The effective-learning-rate confound, and why there is no excuse for it at this budget

Weight normalization makes the effective step size on a normalized parameter scale like `η/||w||^2`. Every "normalized vs unnormalized" comparison in the PEFT literature — LoRA vs DoRA, LoRA vs DeLoRA, DeLoRA vs DeLoRA-PC — is therefore *silently also a learning-rate comparison*. A method that "wins" at the baseline's LR may only be winning because normalization handed it a better effective LR, and *Learning Rate Matters: Vanilla LoRA May Suffice* (2602.04998) is the precedent showing this can consume an entire reported gain.

At flagship scale, per-arm LR sweeps cost hundreds of A100-hours and papers skip them. **In this plan they cost 47.5 GPU-h out of 571 — 8.3% of the project, including 14.1 h of sweeps at 9B itself.** There is no excuse, and the paper says so explicitly: every arm in every table received an independent LR sweep of the same size, and the selected LR per (method, rank, model) is reported in an appendix table.

**Protocol.**

| Tier | Grid | Selection substrate |
|---|---|---|
| Tier 0 (ViT-S) | `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` | VTAB-8, 800/200 split, 50 epochs |
| Tier 1 (ViT-B) | `{3e-4, 1e-3, 3e-3, 1e-2}` after Tier 0 rules out the tails | VTAB-8, 800/200, 50 epochs |
| Tier 2V (ViT-L) | `{ViT-B optimum} × {1/3, 1, 3}` — a transfer *check*, not a fresh sweep | VTAB-8, 800/200, 50 epochs |
| Tier 3L (Qwen3.5-4B) | `{5e-5, 1.5e-4, 5e-4, 1.5e-3}` | 5k-example commonsense subsample, 1 epoch, held-out 1k |
| Tier 4L (Qwen3.5-9B) | `{1.5e-4, 5e-4, 1.5e-3}` — narrowed by the 4B result | 5k-example commonsense subsample, 1 epoch, held-out 1k |
| Tier 4G (9B math) | `{Tier-4L optimum} × {1/3, 1, 3}` | 2.5k math subsample, 1 epoch |

Selection is done once at seed 0 and reused across seeds. We additionally report the **selected-LR-vs-rank curve per method, per model scale** as a figure: if DeLoRA-PC's optimal LR is systematically shifted relative to LoRA's, that *is* the mechanism, and the "gain" is an LR effect wearing a geometry costume. **The LR-vs-scale panel is worth its own figure**: the effective-LR story predicts the shift should track `||W0||` statistics, which differ markedly between a 22M ViT-S and a 9B LLM. Rank-transfer of the selected LR is spot-checked at `r=32` (T1.1) rather than assumed. One caveat we will state: the 4B→9B narrowing of the language grid assumes LR transfers across a family, which is more defensible now that both language tiers *are* the same family than it was when the two tiers were different vendors' models — a small, real benefit of the policy.

Companion controls, all inside the main grid at no extra cost:
- **rsLoRA** (`α/√r`, 2312.03732) as a scaling-artifact control. At `r=2, α=8`, `α/r = 4` vs `α/√r = 5.66`; at `r=32`, `0.25` vs `1.41`. If `gain` measured against rsLoRA flattens, the story is scaling, full stop.
- **Matched effective-LR arm**: LoRA re-run with LR rescaled by the mean `1/||u_i||^2` trajectory measured from the DeLoRA-PC run at the same rank. Costs 3 sweeps in T2.3.

---

### 2.3 Tier 0 — the go/no-go screen — **7.9 GPU-h**

**Purpose.** Answer one question for 8 GPU-hours: *is there any per-component-vs-global effect at all, above the seed floor, at low rank?* Everything downstream is gated on this. Tier 0 is deliberately unchanged — a bigger GPU is not a reason to make the cheapest screen more expensive.

**Substrate.** ViT-S/16 in21k (22M, `timm: vit_small_patch16_224.augreg_in21k`), 224px, bf16, batch 64. Peak VRAM 3.1 GB. **VTAB-8** screening subset — one third of the tasks, all three groups: CIFAR-100, Caltech101, DTD, SVHN (Natural); Patch Camelyon, Resisc45 (Specialized); Clevr-Count, dSprites-Loc (Structured).

Timing anchor: ViT-S/16, 100 epochs × 1000 images = 100k fwd+bwd at 4.6 GFLOPs fwd ≈ 1.4 PFLOPs ≈ 70 s/task at realistic MFU. **VTAB-19 sweep = 0.45 GPU-h; VTAB-8 sweep = 0.19 GPU-h; 50-epoch/800-image short run = 0.095 GPU-h.**

| ID | Experiment | Runs | GPU-h |
|---|---|---|---|
| T0.1 | LR calibration: 5 LRs × 3 arms (LoRA, DeLoRA, DeLoRA-PC) at `r=4`, VTAB-8, short | 15 | 1.9 |
| T0.2 | Rank probe: `r ∈ {1,2,4,16}` × 3 arms × 2 seeds, VTAB-8, full | 24 | 4.6 |
| T0.3 | `r=1` noise-floor calibration (3rd seed at `r=1`) + spectrum/effective-rank readout on saved checkpoints | 6 + diag | 1.4 |
| | **Tier 0 total** | **45** | **7.9** |

**The `r=1` control is the most important 1.4 hours in the project.** At `r=1`, per-component strength is *definitionally identical* to a global scalar (§2.1 `test_reduces_to_delora_at_r1`). Any measured accuracy difference at `r=1` is pure seed/measurement noise, and its magnitude **defines the noise floor** against which every other rank's gap is read. A project that reports "+0.6 at r=2" without knowing that its `r=1` null is `±0.9` has reported nothing. The same control is replicated in language at Qwen3.5-4B (T3L.2) — which is a large part of why the powered language tier is at 4B and not only at 9B.

**GATE 0 — the decision.**

- **PROCEED to Tier 1** if: `mean(DeLoRA-PC) − mean(DeLoRA)` over VTAB-8 at `r ∈ {2,4}` is **≥ 2× the `r=1` null spread**, in the same direction at both ranks, *and* the best-tuned LoRA arm has not beaten both of them.
- **PROCEED, REFRAMED** if: the PC-vs-global gap is inside the noise floor **but** best-tuned LoRA beats *both* normalized arms at `r ∈ {2,4}`. Then the paper becomes the negative/rigor result ("normalized-factor adapters do not transfer to vision; the reported gains are LR-scaling artifacts"), Tier 1 runs at reduced scope (drop rsLoRA and DoRA from the 6-rank grid, keep 3 arms → saves ~16 GPU-h), and the 9B tier is retargeted: instead of testing PC-vs-global at 9B, it tests whether the *family* reproduces at 9B at all, which under this branch is the more interesting question.
- **STOP** if: all three arms are within the `r=1` noise floor of each other at every probed rank *and* the effective-rank diagnostic (T0.3) shows LoRA at `r=2` is **not** collapsing onto one dominant direction. That combination means there is no mechanism to find and no gap to explain. Write up Tier 0 as a 4-page workshop note ("a null result on normalized adapters in vision, with the code"), and reallocate the remaining 563 GPU-h to another project in the program. **A stop here costs 8 GPU-h out of a 1000-h ceiling, which is the entire point of building the ladder this way.**

---

### 2.4 Tier 1 — the main powered result: rank × per-component-vs-global on VTAB-1k — **91.8 GPU-h**

This is the statistical core of the paper. Unlocked by Gate 0.

**Benchmark.** VTAB-1k: 19 datasets in 3 groups, 1000 training images each, standard 800/200 split for hyperparameter selection then retrain on all 1000. Report per-task top-1, the three group means, and the overall mean. VTAB-1k remains the single best credibility-per-GPU-hour setting available: **19 independent tasks means a paired Wilcoxon signed-rank test has real power, which an 8-task commonsense average never will** (§2.12 makes this quantitative — the 8-task average has an across-task-heterogeneity floor that no seed count removes).

**Backbones.** Primary rank grid on **ViT-S/16-in21k** (0.45 GPU-h/sweep, 3.1 GB); confirmation at the two extreme ranks on **ViT-B/16-in21k** (`google/vit-base-patch16-224-in21k`, 86M; 1.5 GPU-h/sweep, 4.8 GB LoRA / 11.8 GB full FT). ViT-L/16 is its own gated tier (§2.6).

**Methods in the 6-rank grid (5, plus SVFT as a diagnostic).** Best-tuned LoRA, rsLoRA, DoRA (reference detached implementation), DeLoRA (stock PEFT, global scalar), DeLoRA-PC. All share the forked PEFT data path. Targets: `q,v` (VTAB convention) with an all-linear variant spot-checked at `r ∈ {2,32}`.

**Recipe.** AdamW, per-method-per-rank LR from §2.2, wd 1e-4, cosine schedule, 100 epochs, batch 64, resize-to-224 without crop/flip (structured tasks break under geometric augmentation), bf16 autocast with fp32 normalization, `torch.compile` off (recompiles per rank and eats the savings).

| ID | Experiment | Runs | GPU-h |
|---|---|---|---|
| T1.1 | LR sweeps, ViT-B, VTAB-8, short: 4 LRs × 5 methods at `r=4`, plus 4 LRs × 3 arms rank-transfer check at `r=32` | 32 short | 10.2 |
| T1.2 | **Core rank grid**, ViT-S/16, VTAB-19: 5 methods × `r ∈ {1,2,4,8,16,32}` × 3 seeds, plus SVFT at `r ∈ {2,8}` × 3 seeds | 96 | 43.2 |
| T1.3 | Confirmation, ViT-B/16, VTAB-19: 3 arms (LoRA, DeLoRA, DeLoRA-PC) × `r ∈ {2,32}` × 3 seeds | 18 | 27.0 |
| T1.4 | Non-rank baselines, ViT-B/16, VTAB-19, 2 seeds: linear probe (cached features, 0.2), BitFit (3.0), full fine-tuning (5.0), VPT-Deep (3.2) | 8 | 11.4 |
| | **Tier 1 total** | **154 runs** | **91.8** |

**Characterizing the crossover.** Define `gain(r) = avg(DeLoRA-PC) − avg(DeLoRA)` (the novel-delta curve) and `gain_L(r) = avg(DeLoRA-PC) − avg(best-tuned LoRA)` (the headline curve). Fit both `c·r^{−β}` and `c·exp(−r/r_0)`; report the crossover rank `r*` at which each drops below the `r=1` noise floor. Then discriminate among three candidate mechanisms by regressing `gain` across all (rank × backbone × task × module-type) cells against:
1. the `α/r` vs `α/√r` scaling gap (**artifact hypothesis** — if `gain_L` measured against rsLoRA flattens, the story is scaling);
2. the effective-rank gap from §2.7(c) (**component-collapse hypothesis**);
3. accumulated `||u_i||` growth from §2.7(e) (**implicit-LR hypothesis**).

With 6 ranks × 19 tasks × 3 seeds × 2 backbones the regression has ~680 cells and is more than anecdote. When Tiers 2V/3L/4L land, the same regression gains a **model-scale** covariate spanning 22M → 9B (four vision backbones and two LLMs), which is the version of this analysis that answers "does the effect shrink with scale" — the question every reviewer of a small-model PEFT paper asks and almost none can answer.

**Parameter-matched accounting.** DeLoRA-PC at rank `r` has exactly `r − 1` more trainable parameters than DeLoRA at rank `r` — `(r−1)/(r(d+k))`, i.e. 0.2% at `r=32, d=k=768` and 0.04% at `d=k=4096`. Negligible, but stated, and the `r=1` control makes the comparison exact at the null.

**GATE 1 — decision after Tier 1, pre-registered.** The success criterion is written to the repo with a commit hash and timestamp *before* T1.2 runs.

- **Case A — PROCEED to Tiers 2, 2V, 3L, 4L, 4G (full program, 471 more hours).** `gain(r=2) ≥ 0.5` VTAB-mean points, paired Wilcoxon across 19 tasks × 3 seeds significant at `p < 0.05` after Holm correction over the 6-rank grid, `gain(r=1)` inside the noise floor, and `gain(r)` monotone decreasing.
- **Case B — PROCEED, mechanism + flagship-scale reproduction only (~272 more hours).** `gain` collapses into the noise floor but `gain_L` (vs tuned LoRA) survives at `r ≤ 4`. Then normalization helps and per-component does not; the paper is "normalized-factor adapters transfer to vision, but DoLoRA's novel delta does not survive a tuned baseline." Run T2.1 + T2.3 + T2.4 (50 h), **skip T2.2 FGVC and Tier 2V** (105 h saved), run Tier 3L in full (143 h) and Tier 4L (79 h) — because under Case B the 9B question becomes "does the *mechanism behind the reported +7.63* survive on a current model with a tuned LoRA baseline," which is a genuinely valuable reproducibility result and the single most-cited thing this project could produce. Skip T4L.3 and Tier 4G.
- **Case C — STOP after Tier 1.** Both `gain` and `gain_L` inside the noise floor at every rank. Publish the negative result with the full 154-run table, the LR-vs-rank figure, and the diagnostics. Total spend 100 h. **One exception, and it is deliberate:** run **T4L.2 at `r=2` only, 3 arms × 2 seeds (35 h)** even under Case C. A vision-only null against a claim made in language is a weaker paper than a vision null *plus* a direct 9B test of the same mechanism in language, and 35 h is cheap insurance against the one criticism that would otherwise be fatal.

Estimated probabilities, stated in advance: A ~25%, B ~40%, C ~35%. (Unchanged — neither a bigger GPU nor a newer base model makes the effect more likely to be real.)

---

### 2.5 Tier 2 — mechanism, backbone dependence, and few-shot transfer — **72.5 GPU-h**

Unlocked by Gate 1 Case A (all of it) or Case B (T2.1 + T2.3 + T2.4 only).

| ID | Experiment | Runs | GPU-h |
|---|---|---|---|
| T2.1 | **Backbone-dependence probe.** 3 arms × 3 backbones × `r=2` × 2 seeds × VTAB-19. Backbones: DINOv2 ViT-B/14 (self-supervised), CLIP ViT-B/16 (contrastive), ViT-B/16-in1k AugReg (supervised, no 21k). | 18 | 29.0 |
| T2.2 | **FGVC 10-shot transfer.** CUB-200, Oxford Flowers-102, Stanford Dogs, Stanford Cars, NABirds; 3 arms × `r=2` × 5 seeds (shot-sampling variance dominates method differences, hence 5). ViT-B/16. | 15 sweeps × 5 sets | 22.5 |
| T2.3 | **Mechanism ablations**, ViT-B/16, VTAB-19, `r=2` × 3 seeds: (a) hard-renormalize `u,v` to unit norm after every optimizer step; (b) weight decay on `u,v` ∈ {0, 1e-4}; (c) `--strict-nonneg` scales; (d) `||W0||` folding on/off. | 12 | 18.0 |
| T2.4 | Checkpoint diagnostics (§2.7), inference-only on adapter checkpoints saved every 5 epochs. | — | 3.0 |
| | **Tier 2 total** | **45** | **72.5** |

**T2.1 is a genuinely new question and the best-value item here.** DeLoRA folds `λ||W0||` into the update; DeLoRA-PC does not. Self-supervised and contrastive backbones have very different weight-norm distributions across layers than supervised ones. If normalized-factor adapters help *because* of base-weight scale heterogeneity, the effect should track the heterogeneity — measurable directly as the across-layer coefficient of variation of `||W0||_F` per backbone, which we compute for free and plot `gain` against. **With the language tiers in scope this diagnostic extends across modalities: Qwen3.5-4B's and Qwen3.5-9B's across-layer `CV(||W0||_F)` are computed on the downloaded checkpoints for zero GPU-hours and enter the same plot as a fifth and sixth point.** The 9B point needs one extra care the previous revision did not: its two block types have different weight-norm distributions, so `CV(||W0||_F)` is reported **within block type** as well as pooled, and both are plotted. If the regression line drawn through four ViT backbones predicts the language points, that is the strongest form this mechanism claim can take.

**T2.3(a) is the ablation that can kill the mechanism claim.** Because `∇_{u_i} L` is orthogonal to `u_i`, `||u_i + ηg||^2 = ||u_i||^2 + η^2||g||^2` grows monotonically, so the effective per-component LR decays like the inverse of accumulated gradient energy — an Adagrad-like schedule nobody configured. Hard-renormalizing after each step removes the norm growth while keeping the geometry exactly. **If the advantage disappears under hard renormalization, the mechanism is implicit LR decay and the geometry story is wrong** — and we say so. Three sweeps, 4.5 GPU-h, and it is the single highest-information ablation in the plan.

---

### 2.6 Tier 2V — ViT-L/16, scale in vision, and the `p=1024` Bessel path — **82.8 GPU-h**

Unlocked by Gate 1 Case A only.

**Why this is here.** In the 3090 plan ViT-L/16 was cut for two reasons and both have dissolved. (i) **VRAM**: ViT-L LoRA at batch 64 / 224px is ~10 GB and full fine-tuning is ~18 GB — both were awkward on 24 GB alongside anything else and are trivial on 48 GB. (ii) **The `p=1024` Bessel-underflow boundary**: ViT-L's hidden width puts the modified-Bessel order at `p/2 − 1 = 511`, exactly where `scipy.special.ive` stops being trustworthy. **That was never a reason to avoid the backbone — it is a reason to use the repo's own CUSF log-Bessel path**, which exists precisely for this regime and has never been exercised on a real workload. Running ViT-L is how the directional-statistics claim in §2.7(f) stops being a `p≤768` convenience result.

**Timing.** ViT-L/16 is 61.6 GFLOPs fwd vs ViT-B's 17.6, i.e. **3.5× ViT-B**: VTAB-19 LoRA sweep **5.3 GPU-h**, full-FT sweep **8.8 GPU-h**, VTAB-8 short run **1.1 GPU-h**.

| ID | Experiment | Runs | GPU-h |
|---|---|---|---|
| T2V.1 | LR transfer check from ViT-B: 3 LRs × 3 arms at `r=2`, VTAB-8, short | 9 short | 9.9 |
| T2V.2 | 3 arms (LoRA, DeLoRA, DeLoRA-PC) × `r ∈ {2,32}` × 2 seeds, VTAB-19 | 12 | 63.6 |
| T2V.3 | Full fine-tuning reference, ViT-L, VTAB-19, 1 seed | 1 | 8.8 |
| T2V.4 | `p=1024` directional statistics through the CUSF / torch log-Bessel path (inference-only on saved checkpoints) | — | 0.5 |
| | **Tier 2V total** | **22** | **82.8** |

**What T2V buys scientifically.** Three vision widths — `p = 384` (ViT-S), `768` (ViT-B), `1024` (ViT-L) — plus `2560` (Qwen3.5-4B) and `4096` (Qwen3.5-9B) gives a five-point width axis for the claim that `gain(r)` shrinks as the base model gets wider (the "a bigger model's spectrum is already rich enough" hypothesis). Two seeds is deliberately thin; T2V is a **trend** measurement whose error bars come from ViT-S and ViT-B, not a standalone significance test, and the tables say so.

**T2V.3 is now the only full-fine-tuning reference above 100M parameters in the whole plan**, because language full-FT no longer fits (§2.9). That raises its value slightly and the paper leans on it accordingly.

**Numerical note, load-bearing for §2.7(f).** At `p = 1024` the Bessel order is 511 and `ive` underflows to zero for the `κ` values we expect (`κ/p ≈ 0.3–1`, so `κ ≈ 300–1000`). T2V.4 runs every `Rbar`/`κ_hat` computation through **both** `scipy.special.ive` and `src/cusf_cpu.py: iv_log` / `src/vmf_kappa_torch.sra_kappa_torch`, and reports the disagreement as a figure. This is a small, honest, reusable contribution: an empirical map of where the SciPy path silently fails on real adapter data.

---

### 2.7 Diagnostics: what we measure on saved checkpoints — **4.9 GPU-h total, inference-only** (folded into T0.3 / T2.4 / T2V.4 / T3L.2 / T4L.2)

Each diagnostic is a falsifiable prediction, not a plot.

**(a) The scale-coupling claim, tested directly.** LoRA's `ΔW = BA` is invariant under `(B,A) → (cB, A/c)`. The generator of that orbit is `(B, −A)`, so the fraction of the gradient that does nothing to `ΔW` is

```
g_inv = ( ⟨∇_B L, B⟩ − ⟨∇_A L, A⟩ ) / ( ||∇_B L||_F^2 + ||∇_A L||_F^2 )^{1/2}
```

For DeLoRA-PC this is **exactly zero by the tangent projection** (proved analytically by the §2.1 test). Measuring `|g_inv|` for LoRA over training, per layer, per rank, is a two-line computation and is the sharpest available test of "removal of scale-dependent gradient coupling." Prediction: `|g_inv|` is largest at small `r` and grows over training. If it does *not* grow, the paper's stated mechanism is wrong and we say so.

**(b) Norm-imbalance drift.** Per rank-one component, track `log(||a_i||/||b_i||)` for LoRA; report the across-component/across-layer variance of this log-ratio vs step. Report the companion for DeLoRA-PC — where the ratio is definitionally irrelevant to `ΔW` — as the "wasted parameter motion" fraction.

**(c) Effective rank of `ΔW`.** Never form the `d×k` matrix: QR the two factors (`O(r^2(d+k))`), SVD the `r×r` core. This is what makes the diagnostic free at `d = 12288` on Qwen3.5-9B MLP layers. Report **stable rank** `||ΔW||_F^2/||ΔW||_2^2` and **entropy effective rank** `exp(H(σ/Σσ))`. Prediction that would explain the whole story: LoRA at `r=2` collapses onto ~1 dominant direction while per-component scales keep all `r` alive; the gap closes at `r=32` because LoRA's spectrum is rich enough anyway. If this holds it predicts SVFT (2405.19597, one scalar per rank-one) recovers most of the gain — **SVFT is therefore in the T1.2 grid at `r ∈ {2,8}` as a diagnostic baseline.**

**(d) Learned-scale spectrum.** Full `σ_1..σ_r` per layer at end of training plus the distribution of learned `s_i`. Count "dead" components (`|s_i| < 0.01·max_j|s_j|`) and count sign flips (`s_i < 0`), which the shifted softplus permits and the draft's prose denies. Prediction: DeLoRA-PC at `r=32` prunes itself to an effective rank well below 32 — it is implicitly AdaLoRA — a second candidate explanation for the narrowing gap, free to check. **Checkable at 9B**, where the draft claims the gap has essentially closed at `r=32`; if the dead-component count at 9B `r=32` is high, that is a direct mechanistic explanation of the draft's own rank trend, and one that does not depend on our using the same base model they did.

**(e) The Riemannian statement.** Short proposition: the ambient gradient `∇_{u_i} L = (s_i/||u_i||) P^⊥_{u_hat_i}(·)` equals `||u_i||^{−1}` times the Riemannian gradient of the induced function on `S^{d−1}` in the embedded metric, and the implicit re-normalization at the next forward pass is exactly the projective retraction `R_x(ξ) = (x+ξ)/||x+ξ||`. So Euclidean SGD on `u` **is** Riemannian SGD on `(S^{d−1})^r` with a per-component learning rate scaled by `1/||u_i||^2`. nGPT does this retraction eagerly; DeLoRA-PC does it lazily at read time. Log `||u_i||_t` for every component; T2.3(a) is the experiment that decides whether this is the mechanism or a description.

**(f) vMF / `Rbar` on adapter directions — and an honest statement of what is not measurable.** The `u_hat_i, v_hat_i` are genuine unit vectors, so directional statistics apply — but the repo's own κ benchmark forbids the naive version, and the language tiers make the numerical constraints *more* binding, not less.

- **Dimensions and the estimator path.** `κ/p` governs estimator accuracy, and the Bessel order is `p/2 − 1`. Exact hidden and intermediate widths are read off the released `config.json` at implementation time and re-printed in the appendix; the values below are what we plan against, and the path decision does not change for anything `p ≥ 1024` regardless.

  | Model | `p` | Bessel order | Path |
  |---|---|---|---|
  | ViT-S/16 | 384 | 191 | `scipy.special.ive` safe |
  | ViT-B/16, DINOv2 ViT-B/14 | 768 | 383 | `scipy.special.ive` safe |
  | **ViT-L/16** | **1024** | **511** | **CUSF / torch log-Bessel mandatory** (T2V.4 quantifies the SciPy failure) |
  | **Qwen3.5-4B** (`q_proj`) | **2560** | **1279** | **CUSF / torch log-Bessel mandatory** |
  | **Qwen3.5-9B** (`q_proj`) | **4096** | **2047** | **CUSF / torch log-Bessel mandatory**; SciPy underflows by orders of magnitude |
  | Qwen3.5-9B (`down_proj` input side) | 12288 | 6143 | CUSF only; `Rbar` reported, `κ_hat` reported only with its null |

  Concretely: `src/cusf_cpu.py: iv_log` or `src/vmf_kappa_torch.sra_kappa_torch`. The exact SciPy version and the CUSF build hash go in the reproducibility appendix. **This is the first workload in the program that genuinely requires the log-Bessel path, and that is a reason to run it, not to avoid it.**
- **Sample size still forbids per-layer `κ_hat`.** With `r = 1..32`, a single layer gives `n = r` directions. At `r = 1`, `n = 1` gives `Rbar = 1` and a division by zero in every κ estimator. **Per-layer `κ_hat` on `r` directions is not a measurement and will not be reported at any scale.** Per-input or per-layer statistics, where reported at all, are vMF *likelihoods* under a fitted reference, never estimated κ.
- **What is reported instead.** (i) **Pool across layers within a matched (module-type, dimension) class.** ViT-B `q_proj` across 12 blocks at `r=32` gives `n = 384` in `p = 768` (`n/p = 0.5`). ViT-L gives `n = 768` in `p = 1024` (`n/p = 0.75`). **Qwen3.5-9B gives `n ≈ 32 × (q_proj-bearing blocks) ≈ 1150` in `p = 4096` (`n/p ≈ 0.28`)** — worse, not better, and the paper says so: going to 9B *degrades* the pooled directional statistic even though it improves everything else. The 9B pool is additionally taken **within block type**, never across, since mixing two block types into one vMF pool is a modelling error that would masquerade as low concentration. All are far below the 10–25p rule, so the **mean resultant length `Rbar`** (`src/vmf_kappa.mean_resultant_length`) is the primary statistic and `κ_hat` appears only alongside its null, with the regime (`κ/p`) stated rather than the point estimate. (ii) **Pool across training steps**, which is the quantity we actually care about — drift, not concentration. For each component, `{u_hat_i(t)}` over `T ≈ 20` checkpoints gives a per-component `Rbar` measuring how far it wandered, directly comparable to LoRA (project onto the sphere post hoc). This one does *not* degrade with model size and is the primary geometric readout at 9B.
- **Matched null, mandatory, and the right one for each case.** The finite-sample bias in `Rbar`/`κ_hat` is systematically **upward** and grows as concentration falls, so every reported value gets a null computed with the *same estimator at the same n and the same p*. For pooled-across-layers samples the null is `n` iid uniform unit vectors in `R^p`. For *trajectories* the iid-uniform null is wrong — the sequence is autocorrelated — so the null is an **isotropic random walk on `S^{p−1}` with per-step arc lengths resampled from the observed step-length distribution**, same `T`, same estimator. Observed and null are printed side by side in every table; only the difference is interpretable.

This is, to our knowledge, the first null-calibrated directional-statistics characterization of adapter geometry, it now spans `p = 384 → 12288`, it costs 4.9 GPU-h, and it is what makes the project a paper rather than an ablation table.

---

### 2.8 How to spend the language budget: 9B fidelity vs mid-size power — the argument, with numbers

This is the decision the envelope and the model policy jointly force, and it deserves an explicit answer rather than an assumption. The question is whether to put the language budget into **one subsampled Qwen3.5-9B grid** or into **Qwen3.5-4B with many more seeds**.

**First, the honest statement of what the model policy costs this tier.** The DoLoRA draft's +7.63 headline was measured on **LLaMA-3-8B**, with **Gemma-3-4B** as its second model. Running Qwen3.5-9B instead means T4L.2 is **not a replication of that number**. We cannot place our commonsense average next to their 75.42 → 83.05 and claim to have reproduced or refuted their table row: the base model, its pretraining corpus, its tokenizer, its commonsense priors and its instruction-free base behaviour all differ, so any discrepancy in absolute numbers is uninterpretable. **That is a real loss and the paper says it in the limitations section and in the caption of the 9B table, not only in a footnote.**

What is gained is worth stating with equal precision. (i) A base model that is **Apache-2.0, currently released, and ungated**, so the artifact — adapters, merged-eval harness, and all — is reproducible by any reader without a licence request, which the LLaMA-3 version was not. (ii) A base model that is *current*, so a null result cannot be dismissed as "that finding was about a 2024 model." (iii) **A same-family pair across two scales** (4B and 9B), which makes the 4B→9B LR narrowing (§2.2) and the scale covariate in the §2.4 regression far cleaner than a 1.5B-Qwen / 8B-LLaMA pair ever was — previously those two points differed in vendor, data, and tokenizer simultaneously with scale.

And the claim we can still make is the one that actually matters scientifically. The draft asserts a **mechanism** — per-component strength on normalized rank-one factors — and mechanisms are supposed to transfer. A +7.63 that appears on LLaMA-3-8B and evaporates on a same-scale current model is either a mechanism that does not generalize or an artifact of an untuned baseline, and **either finding is worth the tier**. We therefore relabel this tier **mechanism reproduction, not replication**, everywhere in the plan and in the paper; the abstract does not use the word "replicate"; and the exact-model replication is listed in §2.11 as out of scope by policy rather than pretended away.

**Cost per run, measured in FLOPs and stated transparently.** With gradient checkpointing, LoRA training costs ≈ `6ND` (fwd `2ND`, activation-gradient bwd `2ND`, checkpoint re-forward `2ND`; weight gradients are negligible because the base is frozen). At 35% MFU on an A6000 (54 TFLOP/s effective):

| Config | Tokens | Train | +Eval | **Per run** |
|---|---|---|---|---|
| Qwen3.5-9B, commonsense-20k × 3 ep, cutoff 256, packed | 15.4 M | 4.4 h | 0.6 h (vLLM, 8 tasks) | **5.8 h** |
| Qwen3.5-9B, commonsense-60k × 3 ep | 46 M | 13.0 h | 0.6 h | **16.1 h** |
| Qwen3.5-9B, math10k × 3 ep, cutoff 512 | 15.4 M | 4.4 h | 0.8 h (GSM8K CoT gen) | **6.3 h** |
| Qwen3.5-4B, commonsense-20k × 3 ep | 15.4 M | 1.9 h | 0.35 h | **2.4 h** |

*Sanity checks against the policy's stated substitution factors.* 4B vs a 1.5B model: `1.9 / 0.71 = 2.7×` — inside the stated 2.5–3× band. 9B vs a 7B model: `9/7 = 1.29×` — inside the stated 1.2–1.4× band. 9B vs the previous plan's 8B: 1.15×, which is where the whole Tier-4 inflation comes from.

*Reconciling with the stated calibration.* The envelope's "20k examples ≈ 8–12 h per flagship run" corresponds to **unpacked** sequences at a 512–1024 cutoff, i.e. ~4× more padded tokens than we actually need. commonsense170k items are short (median ≈ 90 tokens with the LLM-Adapters template); **a 256-token cutoff with sequence packing is what makes this tier affordable, and it is the single most important efficiency decision in the plan.** We budget 5.8 h/run rather than the 4.4 h the FLOP count implies, as a ~30% safety margin, and we hold the 8–12 h figure as the fallback if packing turns out to interact badly with the loss mask (R8).

**Power per GPU-hour, computed rather than asserted.** Using across-task heterogeneity `τ` and seed noise `σ` for the paired arm difference, the standard error of a `K`-task average over `S` seeds is `sqrt(τ²/K + σ²/(K·S))`; MDE at 80% power / two-sided α=0.05 is `2.80 ×` that.

| Setting | `K` | Seeds | MDE | Cost |
|---|---|---|---|---|
| VTAB-1k, ViT-S (`τ=0.8, σ=0.7`) | 19 | 3 | **0.58 pts** | 43 h |
| Commonsense, Qwen3.5-4B (`τ=1.0, σ=1.2`) | 8 | 5 | **1.12 pts** | 132 h (4-rank curve) |
| Commonsense, Qwen3.5-9B | 8 | 2 | **1.30 pts** | 70 h |
| Commonsense, Qwen3.5-9B | 8 | 5 | **1.06 pts** | 174 h |
| GSM8K, Qwen3.5-9B (single metric, `sd_diff=1.8`) | 1 | 3 | **2.91 pts** | 57 h |

**The result is not the one the naive "many small runs" heuristic predicts, and that is why it is worth computing.** On the 8-task commonsense average, going from 2 seeds to 5 seeds at 9B improves the MDE only from 1.30 to 1.06 — because with only 8 tasks the **across-task heterogeneity term `τ²/K` dominates and no seed count removes it**. Buying 104 extra GPU-hours for a 0.24-point MDE improvement is a bad trade. The "many seeds at small scale beats two seeds at flagship scale" rule is correct in general but it is *not* correct for a fixed 8-task average, and the honest thing is to say so.

**Therefore both tiers run, with different jobs:**

1. **Qwen3.5-4B is where the seeds go**, because what needs seeds is not the point estimate — it is the **`r=1` null**, the **shape of `gain(r)` across ranks**, and the crossover rank `r*`. Those are 5-seed, multi-rank objects and they cost 132 h at 4B versus ~320 h for the identical grid at 9B. 4B is the smallest policy-compliant model, so it is also the cheapest place this program can put statistical power in language at all — a real constraint, since the previous revision bought the same rank curve for 63 h at 1.5B.
2. **Qwen3.5-9B is where flagship-scale mechanism reproduction goes**, at 2 seeds and 2 ranks, because the effect it is testing is **+7.63**, not +0.5. Against a 1.30-point MDE that is a **5.9σ** claim; two seeds is ample, and pretending we need five is how a plan talks itself out of running the experiment that matters.
3. **GSM8K runs at 9B** and only at 9B. It was cut from the 3090 plan because a sub-1B model's GSM8K accuracy floor (single-digit EM) made a ±2-point effect unmeasurable — that reasoning was correct and is marginal even at 4B. At 9B, DoLoRA reports 53.5 → 61.0 at `r=2`, its **largest single claimed gain**, in the regime where the metric has room. With 3 seeds the GSM8K MDE is 2.91 EM points against a claimed 7.5 — comfortably detectable, and a null there is a strong, publishable refutation of the mechanism. (Absolute GSM8K numbers on a current base model will be higher than the draft's LLaMA-3 numbers; we report the tuned-LoRA-at-9B baseline explicitly and compare *deltas*, never levels.)
4. **Qwen3.5-27B and the 35B MoE are not run.** 27B LoRA in bf16 is **~56 GB and does not fit the A6000 at all**; the 35B MoE is comparable and adds routing/adapter-placement questions that are a different paper. Both are deferred to the reserved 2×A100 node (§2.11). The width axis is already covered by `p = 384/768/1024/2560/4096`.
5. **No full fine-tuning in language, at any tier.** The smallest policy-compliant model is 4B; full FT at 4B is ≈ 64 GB of weights + grads + fp32 Adam before activations and does not fit 48 GB. The previous revision's 1.5B full-FT reference (2.9 h) is therefore **deleted, not re-costed**, and moved to §2.11. This is a genuine loss of a baseline and it is listed as such in §2.12.

---

### 2.9 Tier 3L — Qwen3.5-4B, the powered language rank curve — **143.4 GPU-h**

Unlocked by Gate 1 Case A or B. Purpose: establish in language the two things vision established and 9B cannot afford — **a real noise floor and a rank curve**.

**Model.** Qwen3.5-4B (base, not instruct). bf16 weights ~8 GB; LoRA at `r ≤ 32` on `q,k,v,up,down`; batch 8 × 512 tokens with gradient checkpointing. **Peak VRAM ~12 GB.** Full fine-tuning is **not run** — ≈64 GB, does not fit (§2.8 item 5, §2.11).

**Data.** A **20k-example stratified subsample of commonsense170k** (LLM-Adapters, Hu et al. EMNLP 2023), sampled proportionally across the 8 constituent datasets with a **fixed seed, identical for every arm, rank, and seed** — the subsample is a fixed property of the benchmark, not a nuisance variable, which keeps every comparison exactly paired. 3 epochs, cutoff 256, packed, batch 8 × grad-accum 2. Evaluated on all 8 commonsense tasks with EleutherAI `lm-evaluation-harness` at a **pinned commit hash**, per-task `version` serialized into the results JSON, generative letter-parse protocol (matching LLM-Adapters), `acc` for BoolQ/PIQA/SIQA/WinoGrande/ARC-e and `acc_norm` for HellaSwag/ARC-c/OBQA, footnoted per task. Evaluation runs under **vLLM on merged weights** (`test_merge_matches_vllm` in §2.1 is what makes this safe).

| ID | Experiment | Runs | GPU-h |
|---|---|---|---|
| T3L.1 | LR sweep: 4 LRs × 3 arms at `r=8`, 5k × 1 epoch | 12 short | 11.4 |
| T3L.2 | **Rank curve.** `r=1`: 2 arms (LoRA, DeLoRA≡DeLoRA-PC) × 5 seeds = 10 runs. `r ∈ {2,8,32}`: 3 arms × 5 seeds = 45 runs. Diagnostics (§2.7) on saved adapters included. | 55 | 132.0 |
| | **Tier 3L total** | **67** | **143.4** |

**What was cut here, and why that and not something else.** At 4B a commonsense-20k run costs 2.4 h against 0.9 h at the previous revision's 1.5B, so the untouched tier would have been 189 h. Two cuts, in the order the program's costing rules require — **ranks and steps before seeds**:

1. **`r=4` is dropped from the language rank curve**, leaving `r ∈ {1, 2, 8, 32}`. Four points, log-spaced, still fit both the `c·r^{−β}` and `c·exp(−r/r_0)` forms and still locate the crossover `r*` by interpolation between `r=2` and `r=8`; the six-point resolution of the shape lives in vision (T1.2), which is where the plan always said the shape claim lived. Cost: −36 h. The residual risk is that `r*` falls between 2 and 8, where we now have no interior point — if T3L.2 shows that, the fix is 15 runs at `r=4` (36 h) out of contingency, and that contingency call is pre-registered.
2. **T3L.3, the full fine-tuning reference, is deleted** — not for budget but because it does not fit in 48 GB at 4B. Cost: −2.9 h against the old plan, and a lost baseline.

**Seeds are untouched at 5**, which is the whole point of this tier and the thing the program has repeatedly got wrong before.

**The `r=1` row is the point of this tier.** Ten runs, 24 GPU-h, and they are what license every other number in the language tables. Without them the 9B tier reports a delta with no idea what a null delta looks like in language at all.

**Reading the result.** The sign and rank-ordering of `gain(r)` are what transfer across modalities, not the magnitude. If `gain(r)` is decreasing in `r` and positive at `r=2` in *both* vision and language, that is a modality-independent mechanism claim. If the sign flips, that is also a finding and it directly qualifies the draft's LLM-only evidence — and it would be a reason to reconsider spending Tier 4G. Because 4B and 9B are the same family, a 4B→9B disagreement is now attributable to **scale** rather than to scale-confounded-with-vendor, which is a stronger inference than the previous revision could draw.

---

### 2.10 Tier 4 — Qwen3.5-9B: flagship-scale mechanism reproduction — **172.6 GPU-h**

Unlocked by Gate 1 Case A or B (T4L), and by T4L reproducing the family effect (T4G). Under Gate 1 Case C, only a 35-h stub of T4L.2 runs (see §2.4).

**This is the experiment the 3090 plan declared out of scope, and the largest single line in this plan.** Qwen3.5-9B LoRA in bf16 is ~18 GB of frozen weights, adapters and their Adam states are <150 MB at `r ≤ 32`, and activations at batch 4 × 256 tokens with gradient checkpointing are ~4–5 GB: **~23 GB peak, comfortable in 48 GB.** No quantization, so `||W0||` and its across-layer distribution — the quantity §2.5/§2.7 treat as a candidate mechanism — are the real ones. vLLM evaluation runs after training releases the training process (18 GB weights + KV cache at `gpu_memory_utilization=0.85` ≈ 41 GB peak).

**Read this tier as mechanism reproduction, not replication.** Everything below tests whether the draft's *mechanism* survives on a current Apache-2.0 model at comparable scale. It does not and cannot test whether their specific numbers on LLaMA-3-8B are correct; §2.8 states what that costs us and what it buys.

**Setup.** `Qwen/Qwen3.5-9B` base. Targets `q,k,v,up,down` (LLM-Adapters convention, matching the draft's target set), resolved per block type as described in §2.1 and with the resulting per-block-type adapter placement and trainable-parameter count printed in the appendix. AdamW, cosine, warmup 3%, LR per §2.2, bf16, gradient checkpointing, packed sequences, cutoff 256 (commonsense) / 512 (math).

| ID | Experiment | Runs | GPU-h |
|---|---|---|---|
| T4L.1 | LR sweep: 3 LRs × 3 arms at `r=2`, 5k × 1 epoch | 9 short | 8.9 |
| T4L.2 | **Flagship mechanism test.** 3 arms (tuned LoRA, DeLoRA, DeLoRA-PC) × `r ∈ {2,32}` × 2 seeds, commonsense-20k, all 8 tasks | 12 | 69.6 |
| T4L.3 | **Data-scale check** (gated on T4L.2 showing a significant effect): LoRA and DeLoRA-PC at `r=2`, commonsense-**60k**, 1 seed | 2 | 32.2 |
| T4G.1 | LR spot-check for math: 3 LRs × 3 arms, 2.5k × 1 epoch | 9 short | 5.2 |
| T4G.2 | **GSM8K.** 3 arms × `r=2` × 3 seeds on math10k (LLM-Adapters), GSM8K test EM with a pinned generative protocol | 9 | 56.7 |
| | **Tier 4 total** | **41** | **172.6** |

**What T4L.2 actually tests, stated precisely.** The draft reports LLaMA-3-8B `r=2` commonsense average 75.42 (LoRA) → 83.05 (DoLoRA), +7.63, narrowing to ≈ +1 at `r=32`. We test three nested claims, all of them stated as *deltas within our own run*, never as comparisons of levels against the draft's table:

1. **Does the family reproduce on a current model?** `DeLoRA-PC − tuned LoRA` at `r=2`, against a 1.30-point MDE. A claimed +7.63 that reproduces at even +3 is a strong positive; a reproduction at +0.5 means the original gain was mostly an untuned-baseline artifact — the outcome §2.2 exists to detect and which the LR-vs-rank figure would then explain. A reproduction at 0 is ambiguous between "artifact" and "does not transfer to Qwen3.5", and we will say so rather than pick the flattering reading; the 4B tier and the vision rank curve are what break that ambiguity, which is another reason both are upstream of this one.
2. **Does the novel delta reproduce?** `DeLoRA-PC − DeLoRA` at `r=2`. This is the sub-point question and 2 seeds cannot resolve it; we report the point estimate with a confidence interval, label it underpowered in the table caption, and defer the powered version to Tier 3L and to §2.11.
3. **Does the rank trend reproduce?** Is `gain(2) > gain(32)`, matching both the draft and the vision curve? This is a sign test on a large predicted difference, 2 seeds is adequate, and **it is the most model-transferable of the three claims** — a monotone rank trend is exactly the kind of thing a mechanism should carry across base models, which makes it the headline of this tier rather than the point estimate.

**The subsampling caveat, stated in the paper, not the appendix.** 20k of 170k changes the data regime; absolute numbers will sit below published 170k results and are not comparable to them (which, given the base-model change, they were not going to be anyway). We report the tuned-LoRA-at-20k baseline explicitly so readers can see the offset, and **T4L.3 measures the data-scaling slope directly** by re-running the two extreme arms at 60k. If `gain(r=2)` at 60k is within noise of `gain(r=2)` at 20k, the subsample is validated and we say so; if it shrinks by half, the paper reports an extrapolated 170k estimate with a stated uncertainty rather than a claim. **This is the honest cost of the subsample and 32 GPU-h is a reasonable price for retiring it.**

**Why GSM8K is here.** It was cut at sub-1B scale for a correct reason — the accuracy floor — and that reason does not apply at 9B, where the draft's `+7.5` is its largest claimed effect and the metric has headroom. Note also that GSM8K is a *single* metric with no across-task heterogeneity floor, so seeds buy full power here (MDE 2.91 at 3 seeds vs 3.56 at 2) — the opposite of the commonsense case, which is why this tier gets 3 seeds and T4L.2 gets 2. Reported as strict exact-match on the final numeric answer with the extraction regex pinned in the repo. One base-model caveat we will flag: a current 9B base has seen more math than a 2024 8B base, so its LoRA baseline starts higher and the *headroom* for a +7.5 is smaller; a smaller absolute gain at 9B is therefore not automatically a refutation, and the paper reports the gain both in EM points and as a fraction of the baseline's error rate.

---

### 2.11 Deferred: what still needs hardware beyond one A6000

Each item lists its ask and the result that would unlock it. The reserved **2×A100 (80 GB each)** node is the default target for anything in this table; H100 spot is the fallback where the node is oversubscribed.

| Deferred item | Why it does not fit | Ask | Unlock condition |
|---|---|---|---|
| **Full commonsense170k at 9B** (no subsample) | 70–100 A6000-h *per run*; the 12-run T4L.2 grid becomes 840–1200 h and blows the ceiling by itself | ~70 h on the 2×A100 node for a 4-run confirmation (2 arms × `r=2` × 2 seeds) | T4L.3 shows `gain` *growing* with data — i.e. the subsample is understating the effect and the full corpus is where the claim lives |
| **Powered PC-vs-global at 9B** (5 seeds × 4 ranks) | ~320 A6000-h for the 55-run grid, on top of a tier that already costs 173 h | ~120 A100-h | Tier 3L and T4L.2 agree in sign at `r=2` but the 9B point estimate sits between 0.3 and 1.3 — exactly the ambiguous zone 2 seeds cannot resolve |
| **Qwen3.5-27B dense, LoRA bf16** | **~56 GB for the frozen base alone — does not fit 48 GB.** No quantization allowed (it would change `||W0||`, the quantity under study) | 2×A100, ~200 A100-h for 3 arms × `r=2` × 2 seeds on commonsense-20k | `gain(r=2)` is *not* decreasing across `p = 384 → 4096`; a flat or increasing scale trend makes the large-model point genuinely load-bearing rather than decorative |
| **Qwen3.5-35B MoE, LoRA bf16** | Comparable footprint to 27B, does not fit; also raises adapter-placement-vs-routing questions that are a separate study | 2×A100, ~250 A100-h | Only after the 27B dense point lands, and only if a reviewer or the 27B result makes sparsity the live question |
| **Any full fine-tuning in language** | Smallest policy-compliant model is 4B; full FT ≈ 64 GB (weights + grads + fp32 Adam) before activations | 2×A100 with FSDP, ~40 A100-h for 4B × 2 seeds on commonsense-20k | A reviewer asks for the full-FT ceiling in language, which vision supplies (T2V.3) and language currently does not |
| **A second 9B-class model** | Doubles Tier 4; no new mechanism, only generalization — and the policy restricts the substitute set, so the honest options are within-family | ~200 A100-h | T4L.2 is surprising enough that model-specificity becomes the live question |
| **ViT-H/14 and ImageNet-1k pretraining of any backbone** | ViT-H VTAB-19 sweep ≈ 16 A6000-h; from-scratch pretraining is a different project | ~120 H100-h | Only if Tier 2V shows the vision scale trend is non-monotone |
| **GRPO / RL post-training with DeLoRA-PC adapters** | Fits in 48 GB at 9B (the reference policy is free via adapter-disable, ~18 GB saved) but is a different research question | — | Handled in the RL project, not here; cross-referenced |

**Not deferred — out of scope, two of them by policy and one by method.**
- **Exact replication on LLaMA-3-8B / Gemma-3-4B.** The program's model policy forbids training or finetuning any non-Qwen3.5 base, so the like-for-like replication of the draft's table is not a hardware question and is not on a wishlist — it is simply not this project. §2.8 and the paper's limitations say so plainly.
- **QLoRA at any scale.** It fits, it is cheap, and it would invalidate the mechanism study by quantizing the exact quantity under investigation. If a reviewer asks for 27B, the answer is "on the A100 node in bf16, and here is why NF4 is not the same experiment," not a rushed appendix.

---

### 2.12 Baselines, statistics, power, risks

**Baselines to beat (all with an equal LR-tuning budget, stated explicitly in the paper).** In the rank grid: LoRA with swept LR (2106.09685), rsLoRA (2312.03732), DoRA in its *reference detached* form (2402.09353), DeLoRA from PEFT (2503.18225), SVFT (2405.19597, diagnostic). Non-rank vision baselines: full fine-tuning (ViT-B and ViT-L), linear probe, BitFit, VPT-Deep (2203.12119). Language baselines: tuned LoRA at every rank — **and no full fine-tuning, because it does not fit at 4B or 9B (§2.8, §2.11); this is a real gap in the language tables and is labelled as one rather than quietly omitted.** Discussed and cited but not run, with a one-line reason each: AdaLoRA (2303.10512), VeRA (2310.11454), AdaptFormer (2205.13535), SSF (2210.08823), FacT, ALoRE (2412.08341), Dyn-Tuning (2403.11808), LoRA+ (2402.12354), PiSSA/MiLoRA, LoRA-XS, MoRA, GaLore, SORSA (2409.00055), 1LoRA (2503.08333), BeamLoRA, MAP (2505.23094), StelLA, RiemannLoRA (2507.12142). Frame against the taxonomy in the unified LoRA-variant survey (2601.22708). Cutting 12 baselines is a deliberate trade: **seeds, ranks and scales over method count**, since the novel question is a within-family comparison and a wide baseline table adds citations, not evidence.

**Metrics.** *Vision:* top-1, three VTAB group means, overall mean; FGVC 10-shot means over 5 seeds. *Language:* per-task `acc`/`acc_norm` with the convention footnoted per task, pinned harness commit + per-task `version` in the serialized JSON, vLLM engine version pinned; GSM8K strict exact-match with a pinned extraction regex. *Cost, reported for every method:* peak VRAM, ms/step, tokens/s, and wall-clock-to-target-accuracy alongside epochs-to-target — the normalization overhead is `O(r(d+k))` in theory but unfused in practice, and nGPT's 60–80% per-step penalty is the cautionary precedent. **The draft claims 1.33× faster convergence; that is a wall-clock claim and we measure it as one, at 9B, in T4L.2 — the first time it will have been checked with matched hardware and tuned LRs on both arms, albeit on a different base model than the one the claim was made on.** *Geometry:* `g_inv`, log-norm-ratio variance, stable rank, entropy effective rank, dead-component and sign-flip counts, `Rbar` with matched null, pooled `κ_hat` with matched null and stated `κ/p` regime.

**Statistical power, pre-registered.** Seed counts are unchanged from the previous revision at every tier; the MDEs below are therefore preserved, with the model names and costs updated.

| Claim | Design | MDE (80%, α=0.05) | Claimed effect |
|---|---|---|---|
| **PRIMARY:** `gain(r=2)` in vision | VTAB-19 × 3 seeds, ViT-S, paired Wilcoxon | **0.58 VTAB-mean pts** | — (this is the novel delta; no prior estimate) |
| `gain(r=2)` in vision at ViT-B | VTAB-19 × 3 seeds | 0.58 pts | — |
| `gain(r)` shape, language | 8 tasks × 5 seeds × 4 ranks, Qwen3.5-4B | 1.12 pts per rank | — |
| `gain_L(r=2)` at 9B (family) | 8 tasks × 2 seeds, Qwen3.5-9B | 1.30 pts | **+7.63** on LLaMA-3-8B (5.9σ margin *if the mechanism transfers at full strength*) |
| `gain(r=2)` at 9B (novel delta) | 8 tasks × 2 seeds | 1.30 pts | ~+1 implied — **underpowered, labelled as such** |
| GSM8K at 9B, `r=2` | 3 seeds, single metric | 2.91 EM pts | **+7.5** on LLaMA-3-8B (2.6σ margin, same caveat) |
| ViT-L scale trend | 2 seeds × 2 ranks | trend only, no significance claimed | — |

The two rows with the "if the mechanism transfers" caveat are the honest form of the model substitution: the *margin* is computed against an effect size measured on a different base model, so it bounds our sensitivity, not our expectation. A 9B result of +2 against a 1.30 MDE is a clear positive for the mechanism even though it is nowhere near +7.63, and the paper will read it that way rather than as a 5.6-point shortfall.

Assumptions: across-task sd of the paired difference `τ = 0.8` (vision) / `1.0` (language), per-seed sd `σ = 0.7` (vision) / `1.2` (language), GSM8K `sd_diff = 1.8`. **Every one of these is checked against the empirical `r=1` null spread (T0.3, T3L.2) and the table is re-issued with measured values before Gate 1.** The language `τ`/`σ` were estimated on a different model family and are the shakiest inputs here; T3L.2's 10-run `r=1` null is what replaces them with measurements, and if measured language `τ` exceeds 1.0 by more than 25% the 9B tier's MDE is re-derived and T4L.2's caption updated before the runs go out. If the measured vision `τ` exceeds its assumption by more than 25%, seeds are added at ViT-S (cheap, 0.45 h/sweep) before any downstream tier runs.

**Statistics.** 3 seeds throughout the vision rank grid, 5 for FGVC few-shot and for the 4B language grid, 2 at 9B commonsense, 3 at 9B GSM8K. Mean ± std in the main table, per-seed numbers in the appendix, paired Wilcoxon signed-rank across the 19 VTAB tasks, Holm correction across the 6-rank grid, and an explicit sentence stating that baselines received an equal or larger hyperparameter budget than the proposed method. The `r=1` null spread is printed in every figure as a shaded band, in both modalities; no gap smaller than that band is described as an effect.

**Risks.**

| # | Risk | Likelihood | What the negative looks like | Publishable? |
|---|---|---|---|---|
| R1 | DeLoRA priority (ICLR 2025, in PEFT) | Certain | Reviewer finds it in one search | Yes — cite in the abstract, fork their PEFT code, reduce the method claim to per-component strength and lead with the rank × scale study + diagnostics |
| R2 | **"You didn't replicate the 8B result — you ran a different model"** | **Certain, and no longer retired** | Reviewer holds the draft's table open and finds no row of ours that lines up with it | Partly mitigated, and stated up front: we run a same-scale, current, ungated, Apache-2.0 base with a tuned LoRA baseline the original lacked, and we call the tier *mechanism reproduction* in the abstract, never *replication*. Residual exposure is real — a reviewer who wants the exact row cannot have it, because training a non-Qwen3.5 base is out of scope by program policy (§2.11). The mitigation that carries weight is claim 3 of §2.10: a **rank trend** is what a mechanism should transfer, and we test it |
| R3 | Per-component beats global by < the noise floor | ~45% | The one novel bit is noise | Yes — Gate 1 Case B/C; becomes a tuned-baseline negative result answering DoRA's and DeLoRA's own stated open questions |
| R4 | Normalized adapters give no vision gain at all | ~35% | LoRA ≈ DeLoRA ≈ DeLoRA-PC across 19 tasks | Yes — Gate 1 Case C plus the 35-h 9B stub, so the null is not vision-only |
| R5 | Effective-LR confound explains everything | ~30% | `gain_L` vanishes under per-arm LR sweep at every scale; T2.3(a) removes the rest | Yes, and arguably the *best* outcome — a clean isolation of an artifact across 22M→9B, with the LR-vs-scale figure as the money plot |
| R6 | κ diagnostics under-powered | Known-certain at per-layer scale, and **worse at 9B** (`n/p ≈ 0.28`, and the pool must be split by block type, halving it again) | `κ_hat` on `r` directions is meaningless; `r=1` divides by zero | Not a risk if handled as in §2.7(f): `Rbar` + matched null, trajectory pooling as the primary readout, CUSF path mandatory at `p ≥ 1024`. Becomes fatal only if the paper promises κ and cannot deliver |
| R7 | lm-eval-harness / vLLM version drift, **plus vLLM support for a recently released architecture** | Certain (drift) / ~20% (support) | Absolute numbers do not match anything published; or merged 9B checkpoints do not load in the pinned vLLM build | Pin harness commit, per-task `version`, vLLM version; make the *rank-ordering of `gain`* the claim; `test_merge_matches_vllm` runs on both 9B block types in week 1 and is the early-warning tripwire. Fallback is HF `generate` batched eval at ~2.5× the eval cost (+0.9 h/run at 9B, +11 h over Tier 4), absorbed by contingency |
| R8 | Packing/loss-mask interaction inflates 9B run time to the 8–12 h figure | ~25% | Tier 4L costs ~165 h instead of 79 h | Absorbed: drop T4L.3 (−32 h) and run T4L.2 at `r=2` only (−35 h); the tier still delivers the headline test |
| R9 | Compute overrun | Low-moderate (the substitution added 96 h) | Grid does not finish before deadline | The ladder is the mitigation: Tier 0 (8 h), Tier 1 (92 h) and Tier 3L (143 h) are each independently publishable and strictly ordered; 15% contingency budgeted; Tier 4 is last and severable, Tier 2V is cut first if the calendar slips |

**Kill criterion.** If Gate 0 stops the project, write the 4-page workshop note and reallocate — 8 GPU-h spent. If Gate 1 lands in Case C, publish the 154-run negative result plus the 35-h 9B stub and stop — 135 GPU-h spent. In neither case does the project consume more than 14% of the ceiling.

---

### 2.13 Summary compute table

Single NVIDIA A6000, 48 GB, bf16, PyTorch 2.x, vLLM for LLM evaluation. All trained language models are Qwen3.5 (Apache 2.0).

**Timing anchors.** ViT-S/16 VTAB-19 sweep 0.45 h; ViT-B/16 VTAB-19 sweep 1.5 h (full-FT 2.5 h); ViT-L/16 VTAB-19 sweep 5.3 h (full-FT 8.8 h); VTAB-8 50-epoch/800-image short run = 0.21× the corresponding VTAB-19 full sweep. Qwen3.5-4B commonsense-20k SFT + 8-task eval 2.4 h; Qwen3.5-9B commonsense-20k SFT + eval 5.8 h; Qwen3.5-9B commonsense-60k 16.1 h; Qwen3.5-9B math10k + GSM8K eval 6.3 h.

| Tier | ID | Line item | Runs | GPU-h | Peak VRAM | Gate |
|---|---|---|---|---|---|---|
| — | 2.1 | Reimplementation, PEFT fork, unit suite | — | 0.1 | CPU | — |
| **0** | T0.1 | LR calibration, ViT-S, VTAB-8 | 15 | 1.9 | 3.1 GB | |
| **0** | T0.2 | Rank probe `r∈{1,2,4,16}`, ViT-S, VTAB-8, 2 seeds | 24 | 4.6 | 3.1 GB | |
| **0** | T0.3 | `r=1` noise floor + spectrum readout | 6 | 1.4 | 3.1 GB | |
| | | **Tier 0 subtotal (+ impl = 8.0)** | **45** | **7.9** | | **GATE 0** |
| **1** | T1.1 | LR sweeps, ViT-B, VTAB-8, short | 32 | 10.2 | 4.8 GB | |
| **1** | T1.2 | Core rank grid, ViT-S, VTAB-19, 5+1 methods × 6 ranks × 3 seeds | 96 | 43.2 | 3.1 GB | |
| **1** | T1.3 | Confirmation, ViT-B, VTAB-19, 3 arms × `r∈{2,32}` × 3 seeds | 18 | 27.0 | 4.8 GB | |
| **1** | T1.4 | Non-rank baselines, ViT-B, VTAB-19, 2 seeds | 8 | 11.4 | 11.8 GB | |
| | | **Tier 1 subtotal** | **154** | **91.8** | | **GATE 1** |
| | | **DECISIVE MINIMUM — complete vision paper** | **199** | **99.8** | | |
| **2** | T2.1 | Backbone probe: DINOv2 / CLIP / in1k, 3 arms × `r=2` × 2 seeds | 18 | 29.0 | 5.2 GB | A or B |
| **2** | T2.2 | FGVC 10-shot, 5 datasets, 3 arms × `r=2` × 5 seeds | 15 | 22.5 | 4.8 GB | A |
| **2** | T2.3 | Mechanism ablations (hard retraction, wd, strict-nonneg, `||W0||` fold) | 12 | 18.0 | 4.8 GB | A or B |
| **2** | T2.4 | Checkpoint diagnostics (inference-only) | — | 3.0 | 2 GB | A or B |
| | | **Tier 2 subtotal** | **45** | **72.5** | | |
| **2V** | T2V.1 | LR transfer check, ViT-L, VTAB-8, short | 9 | 9.9 | 10 GB | A |
| **2V** | T2V.2 | ViT-L/16, VTAB-19, 3 arms × `r∈{2,32}` × 2 seeds | 12 | 63.6 | 10 GB | A |
| **2V** | T2V.3 | ViT-L full fine-tuning reference, 1 seed — *the only full-FT above 100M in the plan* | 1 | 8.8 | **18 GB** | A |
| **2V** | T2V.4 | `p=1024` CUSF vs SciPy Bessel comparison (inference-only) | — | 0.5 | 4 GB | A |
| | | **Tier 2V subtotal** | **22** | **82.8** | | |
| **3L** | T3L.1 | LR sweep, Qwen3.5-4B, 5k × 1 ep | 12 | 11.4 | 12 GB | A or B |
| **3L** | T3L.2 | Rank curve: `r=1`×2 arms×5 seeds + `r∈{2,8,32}`×3 arms×5 seeds, commonsense-20k | 55 | 132.0 | 12 GB | A or B |
| **3L** | — | *(T3L.3 full fine-tuning at 4B — **deleted**, ~64 GB, does not fit; see §2.11)* | — | — | — | — |
| | | **Tier 3L subtotal** — *1.5B → 4B, `r=4` cut* | **67** | **143.4** | | |
| | | **DECISIVE + CROSS-MODALITY MECHANISM** | **266** | **243.2** | | |
| **4L** | T4L.1 | LR sweep, Qwen3.5-9B, 5k × 1 ep | 9 | 8.9 | 23 GB | A or B |
| **4L** | T4L.2 | **Flagship mechanism test**, 3 arms × `r∈{2,32}` × 2 seeds, commonsense-20k | 12 | 69.6 | 23 GB (41 GB eval) | A or B (35 h stub under C) |
| **4L** | T4L.3 | Data-scale check, 60k, 2 runs, 1 seed | 2 | 32.2 | 23 GB | T4L.2 significant |
| **4G** | T4G.1 | LR spot-check, math, 2.5k × 1 ep | 9 | 5.2 | 23 GB | T4L.2 reproduces family |
| **4G** | T4G.2 | **GSM8K**, 3 arms × `r=2` × 3 seeds, math10k | 9 | 56.7 | 23 GB (41 GB eval) | T4L.2 reproduces family |
| | | **Tier 4 subtotal** | **41** | **172.6** | | |
| | | **COMMITTED TOTAL (all gates open)** | **374** | **571.1** | **41 GB max** | |
| | | **+ 15% rerun / debugging contingency** | | **+85.7** | | |
| | | **BUDGETED TOTAL** | | **656.8** | | vs. 1000 ceiling |

**What the model policy did to the budget, in one line.** Vision is untouched at 255.1 h. Language went from 219.8 h to 316.0 h (+44%): +73 h from 1.5B→4B on the rank curve (already after cutting `r=4`, which saved 36 h), +23 h from 8B→9B on the flagship tier, −2.9 h from deleting the language full-FT reference that no longer fits. Committed total moves 474.9 → 571.1 h, budgeted 546 → 657 h, and the program goes from 55% to 66% of the 1000-h ceiling. No seed count anywhere was reduced.

**Spend by branch.** Gate 0 STOP: 8 h. Gate 1 Case C: 135 h. Gate 1 Case B: ~372 h. Gate 1 Case A: 571 h (657 budgeted). The program never exceeds 66% of the ceiling and stops at 0.8% in the worst case.

**Wall-clock.** 657 GPU-h is ~27 days of continuous single-GPU time, ~9–11 weeks at realistic utilization including data prep, evaluation, and analysis. The two language tiers are now jointly the long pole: 316 h in ~108 runs, each 2.4–16.1 h, which schedules cleanly overnight but no longer fits in a single spare fortnight the way the old 1.5B tier did. Storage: ViT adapters are 1–3 MB each; Qwen3.5-9B adapters at `r=32` on 5 module types are ~200 MB each and we keep **adapters only, never merged 18 GB checkpoints** — merging happens transiently for vLLM eval and the merged copy is deleted. Total ≈ 55 GB.

**Timeline (ICLR/NeurIPS cycle, starting August).** Weeks 1–2: reimplementation, PEFT fork, unit suite (including the vLLM merge test on **both** Qwen3.5-9B block types — this is the R7 tripwire and it runs in week 1, not week 8), VTAB-1k pipeline, commonsense-20k subsample fixed and committed. Week 3: Tier 0 and the Gate 0 decision (8 GPU-h ≈ 2 days). Weeks 4–7: Tier 1 (92 h), Gate 1 decision, pre-registered write-up of the main vision table. Weeks 8–12: Tier 3L (143 h) — the language rank curve and its `r=1` null; the paper is submittable at the end of this window. Weeks 12–16: Tier 4L then Tier 4G (173 h) overlapping with Tier 2 (73 h) analysis; Tier 2V (83 h) runs last and is the first thing cut if the calendar slips. Weeks 16–18: diagnostics, the scale × rank mechanism regression, and the paper. **The ordering is deliberate: every tier is severable from the right, and the three most decisive tiers (0, 1, 3L) are also the three earliest.** The honest schedule cost of the policy is roughly two weeks, almost all of it in Tier 3L.


---

## Project 3 — "Conditioning, Not Capacity": Does Per-Component Normalization Help More When the Learning Signal Is One Bit?

**Status: in scope as a real RL study, now on Qwen3.5 and on a defensible seed count.** The earlier re-scope cancelled the RLVR-on-code program twice over — once on budget (~12,600 H100-hours), once on memory (a 1.5B policy with colocated rollout does not train usefully in 24 GB) — and replaced it with a vision bandit proxy plus a 0.5B "cheapest pilot that exists." The corrected envelope removes the memory objection entirely. **Post-training runs on an NVIDIA A6000, 48 GB**, and with LoRA the base model is frozen: no base gradients, no base optimizer states, and — decisively for RL — **no second copy of the model for the reference policy**, because disabling the adapter *is* `π_ref` (~18 GB saved at 9B). Qwen3.5-9B LoRA GRPO fits in 48 GB with a colocated paged-KV rollout engine, tightly. The binding constraint is **wall-clock, not VRAM**.

Two things changed in this revision, and the second is the important one.

**Models.** Everything trained here is **Qwen3.5** (Apache 2.0): **Qwen3.5-4B-Instruct** at Tier 1 and **Qwen3.5-9B-Instruct** at Tier 2. There is no Qwen3.5 "Coder" variant and none is needed — the Qwen2.5 → Qwen3.5 generational jump is large enough that the general models clear the old Qwen2.5-Coder checkpoints on MBPP-Plus and HumanEval+ outright. **Instruct rather than Base**, for one reason: GRPO needs the policy to emit parseable, format-compliant completions from step 0, or the dead-group fraction starts near 1 and the `s = 0` escape pathology (§3.2) becomes inseparable from format failure. 27B dense and the 35B MoE do **not** fit 48 GB under LoRA (~56 GB) and are out of this project's envelope entirely; they are not deferred asks either, because a third scale point is not a new claim (§3.10).

**Seeds.** The previous revision fixed an underpowered design by buying 12 seeds. That was the wrong instrument. **Twelve seeds is far outside normal practice; 3 is the field norm and 5 is generous**, and a design that needs 12 to say anything is a design whose analysis is wrong, not whose budget is small. This revision runs **5 seeds at 4B and 3 at 9B**, and recovers the resolution from replication that is cheaper and more standard than seeds: common-random-number pairing across arms, pooling across `r ∈ {1,2}`, and a per-problem mixed-effects / paired-bootstrap analysis over MBPP-Plus's 378 problems instead of treating one run-level mean as one observation. §3.6 does this honestly, including the part where per-problem pooling **does not** help as much as the VTAB-19 analogy suggests, and the consequence: **the pre-registered primary claim is now the simple effect, not the interaction.**

**Total: ~656 GPU-hours on one A6000** (66% of the 1000-hour ceiling), expected ~186 GPU-h given honest gate priors. That is essentially the previous total (662 h) spent on models 2.5× larger — the seed reduction paid for the model upgrade almost exactly. The vision proxy survives as **Tier 0**, the mechanism experiment.

**Naming**, consistent with Project 2: the draft's "DoLoRA" is renamed **DeLoRA-PC** (per-component). Its only genuine delta over DeLoRA ([arXiv:2503.18225](https://arxiv.org/abs/2503.18225), Bini/Girrbach/Akata, ICLR 2025, merged as `peft.tuners.delora`) is per-component scales `S = diag(s_1..s_r)` versus DeLoRA's single learnable boundary scalar `λ` per layer. Implementation, unit tests, and the PEFT fork all come from Project 2 §2.1 at **< 0.1 GPU-h**; nothing here re-pays that cost. DeLoRA-PC's own Appendix A proves a global scale is *strictly worse* when the target's singular values are non-uniform — the theoretical case against the published baseline, stated by a paper that never ran the comparison. **The sharp question of this entire line is "does per-component strength beat one global scale?"**, and DeLoRA (single scalar) is therefore a first-class arm at every tier, not an ablation.

---

### 3.1 What changed, and what did not

| Item | Previous version | This version |
|---|---|---|
| Trained policies | Qwen2.5-Coder-1.5B / -7B | **Qwen3.5-4B-Instruct (Tier 1) / Qwen3.5-9B-Instruct (Tier 2)** |
| Seeds at LLM scale | 12 at 1.5B, 5 at 7B | **5 at 4B, 3 at 9B** |
| Unit of replication | run-level benchmark mean | **per-problem paired deltas (378 problems) × 2 ranks × seeds, CRN-coupled arms** |
| Primary estimand | interaction `Δ_RLVR − Δ_RFT` | **simple effect `Δ_RLVR` at low rank**; interaction demoted to secondary/exploratory |
| Primary MDE | 1.6 pts (interaction, 12 seeds) | **1.4 pts (simple effect, 5 seeds, pooled ranks)**; interaction 1.6–2.2 pts, reported as underpowered |
| LLM rank grid | `r ∈ {1,2}` crossed + `{4,8}` extension | `r ∈ {1,2}` crossed + **`r = 8` spot check only** (`r = 4` cut) |
| Reference policy / KL | adapter-disable, free | unchanged; **~18 GB saved at 9B**, the reason 9B RL fits |
| Vision tier | 8 seeds at go/no-go | **5 seeds**; design otherwise unchanged, 48.1 GPU-h |
| Analysis checkpoints (Pythia, released-model sweeps) | — | **unaffected by the model policy** — those sections measure released checkpoints and deliberately keep generational diversity |
| Total | 662 GPU-h | **656 GPU-h** |

**Still out, and why.** Pre-training from scratch remains capped at ~100M parameters; nothing here violates it. Full finetuning caps near 1.5–3B on this card (weights + grads + fp32 Adam ≈ 16 bytes/param → 4B full-FT needs ~64 GB), so the full-FT anchor is out. **27B dense and 35B MoE LoRA are out on memory** (~56 GB, needs 2×A100, which is reserved for later scale demonstration). Long-horizon RL (2048-token completions, `G = 16`, full LiveCodeBench-v6 window) fits VRAM but not time. All go to §3.10 with explicit asks.

### 3.2 The hypothesis, decomposed

Let `Δ(r, objective) = score(DeLoRA-PC, r) − score(LoRA, r)` at matched rank, matched data, matched optimizer steps, matched generated-token budget, and **per-arm-per-rank-per-objective tuned learning rate**.

- **H1s (simple effect — the pre-registered primary claim).** `Δ(r, RLVR) > 0` for `r ≤ 2`. Does the normalized, per-component adapter beat plain LoRA under a verifiable-reward objective at low rank? Powered at 4B to 1.4 points (§3.6).
- **H1x (interaction — secondary, exploratory, explicitly underpowered).** `Δ(r, RLVR) > Δ(r, SFT)`. This is a difference of differences and carries roughly 4× the variance of a single arm mean; at 3–5 seeds it resolves to 1.6–2.2 points against a modal predicted effect of `+1.5…+4`. It is reported with a CI and a stated MDE, it is **not** a gate, and no claim in the abstract rests on it. A well-powered simple effect with a matched SFT control reported alongside is a better paper than an underpowered interaction dressed up as the headline.
- **H1a (dose–response — the mechanism evidence).** `Δ(r, RLVR)` grows as the group-relative estimator gets noisier: smaller group size `G`, higher dead-group fraction. Cheap and fully crossed in vision; sampled at 4B (`G ∈ {4, 8, 16}` on the two core arms, inside the LR-calibration budget).
- **H1b (sparsity vs. corruption).** The effect is driven by *information sparsity* (1 bit per rollout), not merely by gradient corruption. Symmetric label noise on dense CE is the vision control. If label noise reproduces it, the story is "any noisy gradient" — weaker, different, and reportable.
- **H2 (variance).** Across-seed std of final score is lower for DeLoRA-PC than LoRA at `r ≤ 2`. **This hypothesis has moved to vision**, where 19 VTAB tasks × 3 seeds give 38 df and detect a 1.55× std ratio at 80% power. At the LLM tier, 5 seeds detect only a **4.4× std ratio** by the naive F-test — useless — so the LLM version is reported as a pooled-over-problems-and-ranks variance-ratio statistic (≈1.5× std ratio, seed-clustered) and is descriptive, not confirmatory. Saying which tier can and cannot test this is the point; inflating seeds to pretend otherwise is what the previous revision did.
- **H3 (per-component vs. global scale).** DeLoRA-PC beats stock DeLoRA. The only architecturally novel question in the line, and a Gate 1 blocker. Under a sparse objective the singular spectrum of the useful update should be *more* non-uniform (few directions carry signal), so H3 and H1s predict the same sign for the same reason — which makes their conjunction a genuine prediction rather than two hopes.

**Why H1s is not obvious in either direction.** *Against:* LoRA Without Regret (Schulman / Thinking Machines 2025, doi 10.64434/tml.20250929) argues a policy gradient absorbs ~1 bit per episode against `O(tokens)` bits for SFT, and finds LoRA matches full FT in RL at rank 1. If capacity is not binding at `r = 1`, a better-conditioned rank-`r` parameterization has nothing to buy and the correct prediction is `Δ ≈ 0` everywhere. *For:* the DeLoRA-PC gradient

```
∇_{u_i} L = (α/r)(s_i/‖u_i‖)(I − û_i û_iᵀ)(∇_W L) v̂_i
```

is an exact orthogonal projection onto the sphere's tangent space. It deletes the radial component — the component that in plain LoRA inflates `‖B‖_F`, and hence the *effective* step size, without changing the direction of `ΔW`. Under a dense gradient the radial drift is mostly signal; under a z-scored group-relative advantage with `G = 8` it is mostly noise, and it compounds over 600 steps. That argument is about estimator variance and is testable at every tier.

**The `s = 0` trap, which must be actively tested.** At init `λ_i = 0 ⇒ s_i = 0`, and the same `s_i` multiplies `∇_{u_i} L`: **the factor directions receive exactly zero gradient at initialization.** Until `s` escapes zero, DeLoRA-PC is a random-feature adapter with learned per-component gains and Kaiming-frozen directions. Under dense CE `s` escapes in tens of steps. Under RLVR with a high dead-group fraction, escape can be far slower, and DeLoRA-PC could *lose* early and never recover — falsifying H1s for a reason unrelated to conditioning. Fully instrumented in §3.7, with pre-registered mitigations, and characterized in vision (where the dead-group fraction is a knob) before a single LLM run.

### 3.3 Tier 0 — the vision bandit proxy, kept as the mechanism experiment

**Why it survives even though real RL is affordable.** It is the only place in the program where the objective's *information content* is a controlled experimental axis with everything else — backbone, adapter, data, steps, batch size — held byte-identical. In an LLM you cannot hold the objective's information content fixed and vary nothing else; in vision you can, and the entire noise dose–response costs single-digit hours. It is also the only place where the `G` sampled actions share **one forward pass**: `G = 8` costs one softmax draw, not eight generations, so the bandit arm costs the same per step as the CE arm. That asymmetry is why a four-level `G` sweep crossed with four adapter arms is 4.7 GPU-h here and would be ~500 GPU-h at 4B.

**Substrate.** ViT-B/16 pretrained on ImageNet-21k (`vit_base_patch16_224.augreg_in21k`, 86M params — inside the ≤100M pre-training cap, and used frozen regardless), adapters on `{q,k,v,proj,fc1,fc2}` of all 12 blocks. `Σ(d_in + d_out)` per block `= 4×(768+768) + (768+3072) + (3072+768) = 13,824`, so **165,888 trainable parameters per unit of rank**: `r=1 → 0.166M` (0.19% of the backbone), `r=32 → 5.3M`. DeLoRA-PC adds `72r` scalars (+144 at `r=2`, 0.04% of the LoRA count); DeLoRA adds 72; LoRA+diag adds `72r`. Publish the exact parameter table. This tier is *analysis-adjacent* substrate, not a language model, and is deliberately left on a released ViT for continuity with Project 2.

**Datasets.** **VTAB-1k** (19 tasks × 1000 images, 800/200 for tuning then retrain on 1000). Dense grids on **CIFAR-100-1k** (100 classes, low random-success rate) and **SVHN-1k** (10 classes, high random-success rate); the 19-task sweep is the confirmation, and it is the template this section copies at the LLM tier: 19 tasks of replication is what lets 3 seeds reach an MDE of ~0.58.

**The objective axis.** Same backbone, adapter, data, steps, batch:

1. **Dense CE** — full label, `log C` bits, low-variance gradient. The SFT analogue.
2. **Dense CE + symmetric label noise** at 20% / 40% — corrupted but dense. The H1b control.
3. **Bandit / verifiable-reward** — treat the model as a policy over classes. Sample `G` labels from `softmax(logits/τ)`, receive **only** `r_g = 1[ŷ_g = y]`, form `A_g = (r_g − mean_g r)/std_g r` exactly as GRPO does, take `∇ = −Σ_g A_g ∇ log π(ŷ_g|x)`. Dead groups dropped as in DAPO's dynamic sampling, dropped fraction **logged, not assumed**. This is the RLVR estimator with the language stripped out: terminal binary reward, no critic, group-relative baseline, z-scored advantage, dynamic sampling.

**Adapter arms (4, all first-class, at every tier).**

| Arm | Update | What it isolates |
|---|---|---|
| **LoRA** | `ΔW = (α/r) B A`, `B = 0` init | Baseline, LR tuned per arm × rank × objective |
| **LoRA + diag scale** | `ΔW = (α/r) B diag(s) A`, `s_i = softplus(λ_i) − log 2`, no normalization | Killer ablation: is the gain just a learnable per-component gain? |
| **DeLoRA** | PEFT stock, per-rank-one normalization, **single** scalar `λ` per layer | The actual prior art (2503.18225) |
| **DeLoRA-PC** | normalized factors, **per-component** `s_i` | The only novel delta (H3) |

**Honest scope of the proxy.** It reproduces the *estimator* — 1-bit terminal reward, group-relative advantage, z-scoring, dead groups, controllable success rate. It does **not** reproduce sequential credit assignment, exploration over a combinatorial output space, or length bias. Tiers 1 and 2 are real code RL, which is strictly better external validity for less argument.

### 3.4 Tiers 1 and 2 — real RLVR on code

**Policy models.** **Qwen3.5-4B-Instruct** (Tier 1) and **Qwen3.5-9B-Instruct** (Tier 2), adapters on all linear layers. The 9B is a hybrid stack (interleaved Gated DeltaNet and gated-attention layers); this changes nothing about the adapter method — every linear projection takes an adapter as usual — and matters in exactly one place, the vMF pooling rule in §3.7, where modules of different type must not be pooled even at equal width.

**A competence problem the model upgrade creates, and the fix.** These are strong general models: 4B-Instruct and 9B-Instruct both sit high on MBPP-Plus out of the box (roughly 70–80% and 80–88% pass@1). That is *worse* for this experiment than the old Qwen2.5-Coder band, because a high base pass rate means (i) little headroom and ceiling-compressed effect sizes, and (ii) a dynamic-sampling dead-group fraction dominated by all-correct groups, which starves the gradient and confounds the `s = 0` escape test. Two pre-registered responses, both decided before any run: **the training prompt mix is shifted hard toward the TACO-verified / PrimeIntellect hard slice and LiveCodeBench-v6, targeting a base pass rate of 40–60% on the training distribution**; and **MBPP-Plus is reported both whole and on a pre-registered difficulty stratum** — the subset the *base* model fails at `n = 16`, a partition defined by the base model alone and therefore identical across arms and not a post-hoc selection. Both strata are reported regardless of which one looks better.

**Algorithm.** GRPO with clip-higher, dynamic sampling (drop all-correct/all-wrong groups), token-level loss normalization, `β = 0`. `G = 8`, 8 prompts/step, `max_new_tokens = 512`, 600 steps. `β = 0` costs nothing in fidelity and buys the KL diagnostic for free, because with LoRA **disabling the adapter is exactly `π_ref`** — no second model, no ~18 GB at 9B, and KL is measurable at every checkpoint without ever being optimized against.

**Data and reward.** MBPP train split (374) plus the hard slice of the TACO-verified / PrimeIntellect verifiable set, ~3k prompts with ≥5 tests each. **60% of each problem's tests are visible to the reward; 40% are held out for final scoring** — the reward-hacking measurement, kept intact. Sandbox: `nsjail`, no network, read-only FS, 6 s/test, 10 s/submission, AST prefilter rejecting test-module imports and `os.system`/`subprocess`/`sys.exit`. **12 pinned CPU workers** — fewer than the 16 the 1.5B plan needed, because a 4B policy generates ~2.4× slower and verification is correspondingly less likely to be the wall-clock bottleneck; re-measure in week 1 and raise if the GPU idles.

**The matched control.** Every GRPO cell has a **rejection-sampling SFT (RFT) twin**: same base, same prompts, same verifier, same generated-token budget, same seed, same data order. Without it, `Δ(RLVR)` alone is uninterpretable as a statement about objectives — it is the control that makes the secondary interaction meaningful and that makes any null interpretable, and it is cheap because the rejection-sampling pool is generated once per base and shared across all arms and seeds.

**vLLM/SGLang is a hard dependency, and here is what it buys.** GRPO at this shape generates `8 × 8 × 512 ≈ 32k` tokens/step. With HuggingFace `generate` and a static KV cache, 4B does roughly 300–400 tok/s at this batch shape → ~90 s/step → **15 GPU-h per run**. With a colocated paged-KV engine (continuous batching, prefix sharing across the `G` rollouts of a prompt — a large win here because all 8 rollouts share the prompt prefix) 4B does ~1,600–2,000 tok/s → ~18 s generation + ~10 s update + verification overlapped on CPU → **~37.5 s/step, 6.25 GPU-h per run**. At 9B the comparison is ~200 tok/s vs ~1,100–1,300 tok/s: **~28 GPU-h/run vs 7.15**. Across the program that is a ~2.5–4× multiplier — without it, Tiers 1+2 cost ~1,600 GPU-h and blow the ceiling. Two further LoRA-specific wins: **only the adapter (tens of MB) is synced to the rollout engine each step**, not the 8/18 GB base, so weight sync is not a per-step tax; and the engine holds exactly one base copy via native multi-LoRA serving. If the stack cannot do colocated vLLM — or cannot serve the 9B hybrid stack's paged KV — the program does not run as written and the fallback is Tier 1 only at 4B. Verify in week 1, not week 6.

**Memory, on the A6000 (48 GB, peak, bf16, verify in week 1 and record):**

| Workload | Peak VRAM | Fits |
|---|---|---|
| ViT-B/16 adapter train, batch 64, 224² | ~9 GB | yes |
| **4B LoRA GRPO** + colocated vLLM (`util = 0.30`), 8×8×(512+512), grad ckpt | ~24 GB | yes, large headroom |
| **9B LoRA GRPO** + colocated vLLM (`sleep_level = 1`), same shape | **~44 GB** (generation phase) | yes, **~4 GB headroom — tight** |
| 9B reference policy | **0 GB** (adapter-disable; ~18 GB saved) | — |
| 9B eval, vLLM only, `n = 16` | ~26 GB | yes |
| 27B LoRA GRPO | ~56 GB | **no** — 2×A100 or out |
| 35B MoE LoRA GRPO | ~56 GB+ | **no** |
| 4B **full** FT (weights + grads + fp32 Adam) | ~64 GB | **no** (full FT caps ~1.5–3B) |

At 9B the split is: training side holds 18 GB frozen base + <0.4 GB adapter/grads/optimizer + ~4 GB activations under gradient checkpointing ≈ 22.4 GB (consistent with the 20–24 GB envelope figure); the rollout engine holds an 18 GB base copy + ~3.5 GB paged KV ≈ 21.5 GB. Peak is the generation phase, ~44 GB, and **4 GB of headroom on a 48 GB card is not comfortable** — this is the one genuine feasibility risk the model upgrade introduces. The pre-registered fallback is `sleep_level = 2` (offload engine weights to host RAM during the update phase), which frees ~18 GB for ~4 s/step, i.e. **+0.7 GPU-h per 9B run, ~+11 GPU-h over Tier 2** — budgeted as a contingency, not in the table. The real memory knob remains `G × completion_length` in the KV cache, which is why 512-token completions are a design parameter here and 2048 is a deferred item.

### 3.5 The ladder and its gates

Every gate names the observation that buys the next tier and the observation that stops the project.

**Tier 0 — is the objective axis real? (5.2 GPU-h, one overnight.)**
CIFAR-100-1k LR calibration (2 objectives × 2 core arms × 5 LRs × 2 seeds), then the go/no-go: CIFAR-100-1k and SVHN-1k × {dense CE, bandit `G=8`} × {LoRA, DeLoRA-PC} × `r = 2` × **5 seeds**.

> **Gate 0 — PASS** requires both: (i) *the axis is real* — the bandit arm's final accuracy is 5–25 points below the CE arm on the same task and adapter (if the gap is <2 points the objective is not information-poor and the proxy is void; if the bandit arm is <2 points above a frozen-adapter baseline it never learned and the proxy is also void); and (ii) *the sign is right* — the paired interaction `Δ_bandit(r=2) − Δ_CE(r=2)` is positive on both tasks with a 95% bootstrap CI whose lower bound exceeds −0.25 points.
> **STOP** if the interaction point estimate is ≤0 on both tasks with the CI upper bound below +0.3. The mechanism is absent in the cheapest, cleanest, highest-powered setting in which it should be most visible. Publish as a short negative note attached to Project 2. **No LLM compute is unlocked from a failed Gate 0.**

**Tier 0B — the mechanism block. (42.9 GPU-h.)**
Rank × objective on 2 tasks (4 arms × 2 objectives × `r ∈ {1,2,8}` × 5 seeds); the full **19-task VTAB-1k** confirmation (2 core arms × 2 objectives × `r=2` × 3 seeds); the noise dose–response (`G ∈ {2,4,8,16}` plus a 2-level dead-group-fraction knob set by initializing the head at low vs. high accuracy — a knob you simply do not have in LLM RLVR); the label-noise control; local LR refinement at the rank extremes.

> **Gate 0B — PASS** requires all four: (a) the interaction is positive on the 19-task sweep, mean `Δ_bandit − Δ_CE ≥ +0.5` points, paired Wilcoxon over tasks, Holm-corrected `p < 0.05` (MDE 0.8 pts at 3 seeds × 19 tasks); (b) **rank-monotone**: interaction at `r=1` > `r=2` > `r=8`; (c) **dose–response**: `Δ` increases monotonically as `G` falls 16 → 2 and as the dead-group fraction rises, Spearman `ρ ≥ 0.8` over the four `G` levels; (d) the LoRA+diag-scale arm does **not** reproduce it.
> **STOP** if (a) fails, or if (c) is flat or non-monotone. A main effect with no dose–response is indistinguishable from a tuning artifact and is not worth 400 GPU-h of RL. If (d) fails — diag-scale matches DeLoRA-PC — the finding is "learnable per-component gain, not normalization": publishable in Project 2, but it does **not** motivate RL compute, because the tangent-projection mechanism is dead.

**Tier 1 — Qwen3.5-4B GRPO: the powered study. (407.8 GPU-h.)**
Four arms at `r = 2` × **5 seeds**; the two core arms at `r = 1` × 5 seeds; an `r = 8` spot check (2 arms × 3 seeds); an RFT-SFT twin for every cell; LR calibration at 40% run length for every arm and objective. All arms within a seed share **common random numbers**: identical data order, identical prompt shuffle, identical rollout RNG streams, identical sandbox seeds, identical eval seeds. Primary benchmark MBPP-Plus (whole and base-fails stratum), co-primary pass@16, secondary HumanEval+ and LiveCodeBench-v6 easy subset (window 2025-06-01 → 2026-05-31).

> **Gate 1 — PASS** requires: (a) **PRIMARY** — the pooled low-rank simple effect `Δ_RLVR = score(DeLoRA-PC) − score(LoRA)` over `r ∈ {1,2}` is positive on MBPP-Plus with a 95% CI excluding zero, by the per-problem seed-clustered paired bootstrap (**MDE 1.4 pts**, §3.6); (b) **H3 holds** — DeLoRA-PC beats stock DeLoRA at `r = 2` under RLVR by a margin whose CI excludes zero (MDE 1.6 pts); (c) the mechanism ordering from Tier 0B reproduces — gradient SNR on direction parameters higher for DeLoRA-PC under RLVR than under RFT; (d) no reward-hacking asymmetry (visible-minus-held-out reward gap differs between arms by <1 point).
> The **interaction** `Δ_RLVR − Δ_RFT` is reported with its CI and its MDE (1.6–2.2 pts) as a secondary, exploratory result. **It does not gate anything**, in either direction, and this is fixed in advance precisely so that a favourable interaction cannot be promoted after the fact.
> **STOP** on (b) failing: if per-component scaling does not beat one global scalar at the scale where the whole claim lives, there is no novel method to confirm at 9B, and the honest output is a rigorous negative on DeLoRA-PC plus a result on *normalization in RL* that belongs to DeLoRA's authors, not to us. **STOP** on (a) failing with a tight CI — a bounded null at 1.4 points is a publishable and useful result, and it does not justify 200 more GPU-hours at 9B.

**Tier 2 — Qwen3.5-9B GRPO: scale confirmation. (200.4 GPU-h.)**
Three arms (LoRA, DeLoRA, DeLoRA-PC) at `r = 2` × **3 seeds**; two core arms at `r = 1` × 3 seeds; RFT twins throughout; 3-point LR calibration at 30% length; same CRN coupling.

> **Gate 2 — PASS** requires the 9B pooled simple effect to be positive at both ranks with a 95% CI excluding zero (**MDE 1.7 pts** — pooling over ranks and problems makes 3 seeds at 9B a real test, which 5 seeds at 7B was not in the previous design), *and* its CI to overlap the 4B estimate. The interaction at 9B (MDE 2.0 pts) is descriptive only.
> **STOP / REPORT** if the sign flips at 9B: "the effect is real at 4B and does not survive to 9B" is the single most interesting negative this program can produce and it should be written as such, with the scale-dependence of the dead-group fraction and the gradient SNR as the proposed explanation.

### 3.6 Statistical power — recomputed at 3–5 seeds, honestly

This is the section the review killed the original design on (`08-review.md` §3.2: interaction MDE 5.5 points at 3 seeds, 4.2 at 5, against a modal predicted effect of ≤2.5). The previous revision fixed it by buying 12 seeds. This revision fixes it by fixing the analysis, and by being explicit about which estimand survives.

**Where the replication actually comes from — including where it does not.** The instruction to copy Project 2's vision tier (MDE 0.58 at 3 seeds, because VTAB-19 supplies 19 tasks) needs one correction before it can be applied, and getting this wrong would be the kind of error this program keeps making in the other direction. **VTAB-19's 19 tasks are 19 independent training runs.** MBPP-Plus's 378 problems are 378 *evaluations of a single training run*. Pooling over problems therefore replicates the **measurement**, not the **training**, and it does **not** shrink the training-seed variance component — which is the dominant one. Four mechanisms are used, and only three of them buy power against seed noise:

1. **Per-problem paired analysis (378 problems).** Kills eval-sampling variance (0.57 pts run-level → ~0.15 pts residual once eval seeds are shared across arms) and conditions the comparison on problem difficulty. Real, but bounded: the seed term is untouched.
2. **Common random numbers across arms.** The largest single lever. Arms within a seed share data order, prompt shuffle, rollout RNG streams, sandbox seeds and eval seeds, which raises the across-arm correlation `ρ` of the training component. The previous design assumed `ρ ≈ 0.4` with loose coupling; tight CRN in stochastic-simulation practice routinely reaches 0.7–0.9. **`ρ = 0.7` is the pre-registered planning target and `ρ = 0.4` the conservative fallback; both MDEs are published below and `ρ̂` is measured from the calibration runs before unblinding.**
3. **Pooling over `r ∈ {1,2}`.** Genuine additional training runs. With a between-rank correlation of the estimates ≈0.5, the effective variance factor is 0.75. The primary contrast therefore rests on **10 paired training runs (5 seeds × 2 ranks), not 3 observations.**
4. **Mixed-effects model.** `pass_{j,s,a} ~ arm × objective × rank + (1|problem) + (1|seed) + (arm|problem)`, fitted on ~3,780 problem-level paired deltas per contrast, with a two-way (problem × seed) cluster bootstrap as the inferential workhorse. This does not beat the seed term either, but the `(arm|problem)` variance component is a *new deliverable* — effect heterogeneity across problems — that no run-level analysis can produce, and it is powered by `J = 378`.

**Variance model at 4B.** Per-seed run-level final MBPP-Plus pass@1 std `σ ≈ 1.7` points, of which eval sampling (`n = 16`, 378 problems, mean `p(1−p) ≈ 0.2`) contributes `σ_eval ≈ 0.57`, leaving `σ_train ≈ 1.60`. Paired difference: `σ_Δ = √(2σ²_train(1−ρ) + σ²_eval,resid)`, giving **1.28** at `ρ = 0.7` and **1.78** at `ρ = 0.4`. The RFT control is quieter (`σ_train ≈ 0.90` → `σ_Δ,RFT = 0.76` / 1.03), so the interaction carries `σ_int = √(σ²_Δ + σ²_Δ,RFT)` = **1.49** / 2.06. Pooling over two ranks multiplies every `σ` by `√0.75 = 0.866`. MDE at 80% power, `α = 0.05` two-sided, is `2.80 × σ/√k`.

| Design | MDE, simple effect `Δ_RLVR` (**primary**) | MDE, interaction `Δ_RLVR − Δ_RFT` (secondary) |
|---|---|---|
| 4B, 3 seeds, single rank, run-level means, `ρ=0.4` (the naive design) | 2.9 pts | 3.3 pts |
| 4B, 3 seeds, pooled `r∈{1,2}`, CRN `ρ=0.7` | 1.8 pts | 2.1 pts |
| **4B, 5 seeds, pooled `r∈{1,2}`, CRN `ρ=0.7` — as budgeted** | **1.4 pts** | **1.6 pts** |
| 4B, 5 seeds, pooled, conservative `ρ=0.4` | 1.9 pts | 2.2 pts |
| 4B, 5 seeds, `r=2` only — the H3 contrast (PC vs. DeLoRA) | 1.6 pts | — |
| **9B, 3 seeds, pooled `r∈{1,2}`, `ρ=0.7` — Tier 2 as budgeted** | **1.7 pts** | 2.0 pts |
| 9B, 5 seeds, pooled (conditional extension, §3.8) | 1.3 pts | 1.5 pts |

**The estimand decision, stated plainly.** After all four pooling mechanisms, the interaction resolves to **1.6 points at best and 2.2 points if CRN under-delivers**, against a modal predicted interaction of `+1.5…+4` (Project 2's Case B, ~55% probability, puts the *SFT* gap at `+1…+2.5`; if the RLVR gap is 2–3× that, the interaction sits in `+1.5…+4`). That covers the upper half of the plausible range and misses the lower half. Chasing it would mean going back to 12 seeds, which is not a design this field accepts. **So the interaction is demoted.** The pre-registered primary claim is the **simple effect** — does the normalized, per-component adapter beat LoRA under RLVR at low rank — at a **final MDE of 1.4 points** (1.9 under the conservative `ρ`), with the RFT twin retained in full so the interaction can be reported as a secondary, explicitly underpowered, exploratory quantity with its CI and its MDE printed beside it. The write-up is pre-committed either way: a primary CI of `[−0.2, +1.2]` is reported as "the low-rank RLVR advantage, if any, is below 1.2 points," which is a real constraint on the claim.

**Why this is not a retreat.** The previous design's headline was an interaction powered to 1.6 points using 12 seeds on a 1.5B model. This design reaches 1.4 points on the primary using 5 seeds on a 4B model, and reaches 1.7 points at 9B on 3 seeds — a scale point the previous design could only treat as a consistency check. The resolution did not come from seeds; it came from pairing, rank pooling, and analysing 378 problems instead of one mean.

**Vision (Tiers 0/0B).** With per-seed per-task `σ ≈ 0.9` points:

| Design | MDE, single `Δ` | MDE, interaction |
|---|---|---|
| 1 task, 5 seeds | 1.6 pts | 2.3 pts |
| **19 tasks paired, 3 seeds — the 0B confirmation** | **0.58 pts** | **0.8 pts** |
| 19 tasks paired, 5 seeds | 0.45 pts | 0.62 pts |

The vision tier resolves effects ~2.5× smaller than the LLM tier at 1/8 the cost, which is the division of labour: **vision establishes the mechanism, its dose–response, and the variance hypothesis (H2, 1.55× std ratio at 38 df); the LLM tiers establish that the mechanism is worth anything on the task the claim is about.**

**Evaluation protocol (all LLM tiers).** `n = 16` samples/problem for GRPO checkpoints (`n = 8` for RFT controls, whose variance is lower), temperature 1.0, top-p 1.0 (no nucleus truncation — it biases pass@k), `max_new_tokens = 1024`. Report the unbiased pass@1 estimator and **pass@16 as a pre-registered co-primary**, because 2504.13837 shows RLVR routinely raises pass@1 while lowering pass@k and a sharper optimizer may trade diversity harder. Final scoring uses **100% of tests**, including the 40% held out from the reward. Checkpoint rule declared in advance (last step primary, best-val secondary); never best-of-run. MDE printed beside every number.

### 3.7 Diagnostics — the actual deliverable

Logged for all arms at identical cadence, at every tier, and all cheap. These, not a leaderboard delta, are what make this a paper — and every one of them has hundreds to thousands of samples *within* each run, so they are powered at 3–5 seeds in a way the benchmark mean is not.

- **Gradient SNR — the direct measurement of the mechanism.** Every 100 steps, freeze parameters and recompute the objective's gradient with `M = 16` independent rollout-sample draws on the same prompt batch. Report `SNR = ‖E ĝ‖² / tr Cov(ĝ)` per parameter group (direction parameters, scale parameters, LoRA `B`/`A`). Prediction: the tangent projection raises SNR on direction parameters under RLVR and much less under RFT-SFT. Note `dL/ds_i = (α/r)⟨∇_W L, û_i v̂_iᵀ⟩` is a scalar contraction against a fixed rank-one direction, hence intrinsically low-variance — the scale parameters should be the *quiet* ones.
- **The `s = 0` escape pathology.** Log `s_i` trajectories, the step at which `max_i s_i` exceeds `10⁻² α/r`, and the cumulative angular movement of each `û_i` from Kaiming init. If directions never move, report it — it reframes the method as a random-feature adapter with learned gains. Pre-registered mitigations, in order: initialize `λ_i = softplus⁻¹(log 2 + s₀)` with `s₀ = 10⁻² α/r`; or a short warm start applied **identically to every arm**. Declare which was used.
- **LoRA radial drift.** `‖B_i‖_F` over training in the LoRA and diag-scale arms — the exact quantity the projection removes. Correlate its growth rate with instability events and with the effect size. If the correlation is absent, the mechanism story is wrong regardless of the accuracy numbers.
- **Dead-group fraction and policy entropy**, every step, both tiers. Report the step at which entropy falls below 50% of initial, per seed. The dead-group fraction is the bridge variable between vision and LLM, and it is the quantity most affected by the move to strong Instruct models (§3.4): if the 4B and 9B runs sit at different points on the vision dose–response curve, that is the pre-registered explanation for any scale disagreement.
- **Reward-vs-KL Pareto.** With LoRA, disabling the adapter *is* `π_ref`, so KL is free to measure at `β = 0`. Plot reward against KL, not against step. **A tie in reward with a significant KL gap is a real, reportable conditioning result** — and it is now measurable at 9B for zero extra VRAM.
- **Reward hacking.** Visible-test minus held-out-test reward per arm per checkpoint; automated AST audit over sampled completions; manual audit of 100 random *passing* completions per arm at the final checkpoint; sandbox violation counts. Both arms share the rollout path, so hacking should be symmetric — verify it, and treat asymmetry as a Gate 1 failure.
- **Effect heterogeneity across problems.** The `(arm|problem)` random slope from §3.6, reported as a distribution: which problems the normalized adapter wins, and whether the win concentrates on the base-fails stratum. This is a deliverable that only exists because the analysis unit is the problem, and it is the scientific dividend of dropping run-level means.
- **Adapter conditioning metrics** on `ΔW` per module, carried from Project 2: 2-norm condition number, full singular spectrum, effective rank (entropy of normalized singular values), mutual coherence `max_{i≠j}|û_iᵀû_j|` for `{û_i}` and `{v̂_i}`.
- **vMF directional statistics, with the repo's estimator and stated honestly.** Fit vMF to pooled unit direction vectors and to per-interval angular displacements between checkpoints. Pooling groups are read from each model's config at run time and the exact table is published; **modules are pooled only when they share both dimension and layer type**, which at 9B means the attention-layer projections and the Gated DeltaNet-layer projections form separate pools even where their widths coincide. Non-negotiable constraints, all inherited from the repo's kappa benchmark:
  1. Every pooled dimension in both Qwen3.5-4B and Qwen3.5-9B is `p ≥ 1024`, so the log-Bessel **must** come from the compiled CUSF or the pure-PyTorch log-Bessel path — `scipy.special.ive` underflows to exactly zero above order ~511, so Sra-with-SciPy returns NaN for *all* of them. There is no dimension in this project where the SciPy path is valid.
  2. With `κ/p` far below 0.05 at these `n`, no achievable `n` buys 1% relative accuracy, so **`κ̂` is never reported as a point estimate.**
  3. Every `κ̂` is reported only as a **matched-null-referenced statistic**: draw `10⁴` sets of `n` uniform unit vectors in `R^p`, run the identical estimator at the identical `n`, report the observed `κ̂`'s empirical quantile in that null. The finite-sample bias is systematically *upward* and grows as concentration falls — i.e. it points the same way as the claim we would like to make, which is exactly why the matched null is mandatory and why an uncalibrated `κ̂` here would be self-serving.
  4. **`n = r` is never an acceptable sample.** With `r ∈ {1,2,8}` adapter directions per module, a per-module `κ̂` is meaningless; pooling across layers within a type-and-width group (and across checkpoint intervals for the displacement statistic) is the only admissible instrument. Any *per-input* directional statistic is a **vMF log-likelihood**, never an estimated `κ`: `n = 1` gives `R̄ = 1` and a division by zero.

### 3.8 Compute budget

**Hardware: one NVIDIA A6000, 48 GB.** All figures are A6000-hours. Calibration: A6000 ≈ 0.35–0.5× an A100 for bf16; the LLM step times below are the low end of that band and should be re-measured in week 1.

**Unit costs, and how the model substitution was priced.** A Qwen3.5-4B GRPO run is costed at **2.5× the old 1.5B run** — the bottom of the stated 2.5–3× band, because a meaningful slice of the 1.5B step time was fixed overhead (verification, adapter sync, optimizer) that does not scale with model size. A Qwen3.5-9B run is costed at **1.3× the old 7B run**, mid-band. One consequence should be stated rather than hidden: **at these multipliers the 4B run (6.25 h) and the 9B run (7.15 h) cost almost the same.** The ladder between Tiers 1 and 2 is therefore much flatter than it was, and Tier 1's justification is no longer "it is cheap" but "it is where the seeds and the ranks are, and it has 24 GB of VRAM headroom for the diagnostics." If week-1 measurement shows 4B closer to 3× (7.5 h/run), Tier 1 grows by ~40 h and the response is the R6 ladder below, not a seed cut.

| Unit | Cost |
|---|---|
| ViT-B/16, one VTAB-1k task (1000 imgs, 100 epochs, batch 64, bf16) + test eval | 0.065 GPU-h |
| **4B GRPO run**: 8 prompts × `G=8` × 512 new tokens, 600 steps, ~37.5 s/step | 6.25 GPU-h |
| 4B GRPO at 40% length (LR calibration) | 2.50 GPU-h |
| 4B RFT-SFT twin (pool shared, matched token budget) | 1.00 GPU-h |
| 4B RFT-SFT twin at calibration length | 0.88 GPU-h |
| 4B eval, MBPP-Plus (378 problems), `n=16` / `n=8` | 1.15 / 0.65 GPU-h |
| 4B full-suite eval (HumanEval+ 164, LCB-v6-easy ~180), `n=16` | 1.25 GPU-h |
| **9B GRPO run**, same shape, ~43 s/step | 7.15 GPU-h |
| 9B GRPO at 30% length | 2.20 GPU-h |
| 9B RFT-SFT twin | 1.30 GPU-h |
| 9B eval, MBPP-Plus `n=16` / `n=8` | 1.50 / 0.80 GPU-h |
| 9B full-suite eval, `n=16` | 1.55 GPU-h |

| Tier | Block | Runs | h/run | GPU-h |
|---|---|---|---|---|
| T0.1 | Vision LR calibration, CIFAR-100-1k: 2 obj × 2 arms × 5 LR × 2 seeds | 40 | 0.065 | 2.6 |
| T0.2 | Vision go/no-go: 2 tasks × 2 obj × 2 arms × `r=2` × 5 seeds | 40 | 0.065 | 2.6 |
| | **Tier 0 subtotal** | **80** | | **5.2** |
| T0B.1 | Rank × objective: 2 tasks × 4 arms × 2 obj × `r∈{1,2,8}` × 5 seeds | 240 | 0.065 | 15.6 |
| T0B.2 | VTAB-1k 19-task confirmation: 2 core arms × 2 obj × `r=2` × 3 seeds | 228 | 0.065 | 14.8 |
| T0B.3 | Noise dose–response: `G∈{2,4,8,16}` (48) + dead-group knob (24) | 72 | 0.065 | 4.7 |
| T0B.4 | Label-noise dense control: 4 arms × {20%,40%} × 3 seeds | 24 | 0.065 | 1.6 |
| T0B.5 | LR refinement at `r∈{1,8}` | 96 | 0.065 | 6.2 |
| | **Tier 0B subtotal** | **660** | | **42.9** |
| T1.1 | 4B LR calibration, GRPO: 4 arms × 4 LR × `r=2`, 40% length | 16 | 2.50 | 40.0 |
| T1.2 | 4B LR calibration, RFT control: 4 arms × 4 LR | 16 | 0.88 | 14.0 |
| T1.3 | **GRPO main, `r=2`: 4 arms × 5 seeds** | 20 | 6.25 | 125.0 |
| T1.4 | **GRPO main, `r=1`: 2 core arms × 5 seeds** | 10 | 6.25 | 62.5 |
| T1.5 | GRPO rank spot check, `r=8`: 2 arms × 3 seeds | 6 | 6.25 | 37.5 |
| T1.6 | Matched RFT-SFT twins, every cell | 36 | 1.00 | 36.0 |
| T1.7 | Shared rejection-sampling pool generation | — | — | 8.0 |
| T1.8 | Eval, GRPO checkpoints, MBPP-Plus `n=16` | 36 | 1.15 | 41.4 |
| T1.9 | Eval, RFT checkpoints, MBPP-Plus `n=8` | 36 | 0.65 | 23.4 |
| T1.10 | Full-suite eval (HumanEval+, LCB-v6-easy) on `r∈{1,2}` cells + base | 16 | 1.25 | 20.0 |
| | **Tier 1 subtotal** | **192** | | **407.8** |
| T2.1 | 9B LR calibration: 3 arms × 3 LR × `r=2`, 30% length | 9 | 2.20 | 19.8 |
| T2.2 | **GRPO, `r=2`: 3 arms × 3 seeds** | 9 | 7.15 | 64.4 |
| T2.3 | **GRPO, `r=1`: 2 core arms × 3 seeds** | 6 | 7.15 | 42.9 |
| T2.4 | Matched RFT-SFT twins | 15 | 1.30 | 19.5 |
| T2.5 | Shared rejection-sampling pool generation | — | — | 10.0 |
| T2.6 | Eval, GRPO checkpoints, MBPP-Plus `n=16` | 15 | 1.50 | 22.5 |
| T2.7 | Eval, RFT checkpoints, `n=8` | 15 | 0.80 | 12.0 |
| T2.8 | Full-suite eval + base | 6 | 1.55 | 9.3 |
| | **Tier 2 subtotal** | **75** | | **200.4** |
| | **TOTAL, all gates passing** | **1,007** | | **656.3** |

**Total: 656 GPU-hours — 66% of the 1000-hour ceiling.** Expected spend given honest gate priors (Gate 0 pass ≈ 0.6, Gate 0B ≈ 0.5, Gate 1 ≈ 0.55) is `5.2 + 0.6×42.9 + 0.30×407.8 + 0.165×200.4 ≈ **186 GPU-hours**`. Wall-clock on one card: ~28 days at full occupancy for the complete program, ~8 days expected. The cheap-first ordering is real — 93% of the budget sits behind two gates that together cost 48 GPU-h to evaluate. Contingency not in the table: `sleep_level = 2` at 9B if the measured peak exceeds ~46 GB (+11 GPU-h).

**Conditional extension (pre-registered, not baseline).** If Gate 1 passes and Gate 2 shows the right sign, raising the 9B seed count from 3 to 5 costs **+113.7 GPU-h** (10 extra GRPO runs, 10 twins, 20 evals, 4 full-suite), bringing the total to **~770 GPU-h** — inside the ceiling — and brings the 9B primary MDE to 1.3 points. Five is the ceiling on that extension, not a waypoint: there is no version of this plan that goes back to 12.

**What was cut relative to the previous revision, explicitly.** (i) **Seeds: 12 → 5 at Tier 1, 5 → 3 at Tier 2.** The resolution is recovered by CRN pairing, rank pooling, and per-problem analysis, not by pretending the old MDE still holds — the primary estimand changed from the interaction to the simple effect precisely because the interaction does not survive the cut (§3.6). (ii) **`r = 4` dropped** from the rank extension; `r = 8` survives at 3 seeds as a monotonicity spot check, because rank-monotonicity is an H1 prediction and vision alone is weaker evidence for it. (iii) **Vision go/no-go 8 → 5 seeds** (−1.6 h), for consistency with the seed-count argument. (iv) The LR grid was **not** cut — 4 points per arm per objective at 4B — because the effective-LR confound is the top-ranked failure mode and cutting it to pay for a model upgrade would be the exact mistake this program keeps making. (v) Carried over from before: the nanoGPT/Countdown bridge; the 0.5B tier; the full-FT anchor; the Olmo-3 second family and the 2k-step GSPO run (deferred, §3.10); DoRA exact-vs-detached (Project 2 unit test, zero GPU); pass@64 and the 4,000-problem eval sweep (reduced to pass@16 on ~720 problems).

**Non-GPU.** 12 pinned CPU workers for the sandbox throughout Tiers 1–2; eval alone submits ~6,000 programs per checkpoint. Storage: VTAB-1k + CIFAR/SVHN ≈ 40 GB; two base models ≈ 26 GB (4B bf16 ~8 GB, 9B bf16 ~18 GB); adapter checkpoints 0.6–90 MB each; rollout logs dominate at ~1.5 TB if every completion is kept, so keep full completions only for the LR-selected cells and store hashes + rewards otherwise (~120 GB). Per-problem eval records (problem × arm × rank × seed × sample) are kept in full — they are the analysis unit and they are small.

### 3.9 Baselines, matching rules, and prior-art positioning

| Arm | Tiers | Ranks | Seeds | Purpose |
|---|---|---|---|---|
| Frozen base / zero-shot | 0–2 | — | — | Absolute floor; confirms RL moved anything |
| Linear probe (vision) | 0–0B | — | 3 | Confirms the bandit arm learned |
| **LoRA** | 0–2 | 1,2,8 | 5 (T0/T0B) / 5 (T1) / 3 (T2) | Primary baseline, LR retuned per arm × rank × objective |
| **LoRA + diag learnable scale** | 0B–1 | 1,2,8 | 5 / 5 | Killer ablation: isolates normalization from the learnable gain |
| **DeLoRA (single scalar)** | 0B–2 | 1,2 | 5 / 5 / 3 | The published prior art; makes H3 the sharp question |
| **DeLoRA-PC** | 0–2 | 1,2,8 | 5 / 5 / 3 | Method |
| Dense-CE twin of every vision arm | 0–0B | same | same | Matched control for the interaction |
| RFT-SFT twin of every LLM arm | 1–2 | same | same | Matched control; retained in full even though the interaction is secondary |

**Matching rules.** Trainable parameters matched to within 0.05% at equal rank (exact table published, read from each model's config). Compute matched on optimizer steps **and generated tokens** — not wall-clock. Arms within a seed are **common-random-number coupled** on data order, prompt shuffle, rollout RNG streams, sandbox seeds and eval seeds; `ρ̂` is estimated from the calibration runs and reported, and if `ρ̂ < 0.55` the published MDEs are revised upward *before* unblinding. **We do not claim a wall-clock speedup**: DeLoRA-PC's `O(r(d+k))` overhead versus DoRA's `O(dk)` is a memory-and-exactness argument, not a throughput argument, and overclaiming it is the easiest thing for a reviewer to falsify. Normalization is computed in fp32 even under bf16 autocast (column norms in wide MLP projections lose ~3 digits in bf16 and the tangent-projection assertion fails spuriously). The Project 2 numerical-equivalence assertion (`max|ΔW_train − BA| < 1e-5`) runs on every checkpoint, so all arms share a byte-identical inference path and only the training graph differs — which is what makes the rollout engine's multi-LoRA serving valid across arms.

**The effective-learning-rate confound.** Weight normalization makes the effective step size scale like `η/‖w‖²`, so *every* normalized-vs-unnormalized comparison is silently also a learning-rate comparison. This was the highest-probability failure mode of the original design. Here: vision gets full-length 5-point sweeps per arm × rank × objective at 4 GPU-minutes each — there is no excuse and no budget argument. At the LLM tiers the committed protocol is **4 LR points at 40% run length at 4B (3 at 30% at 9B), per arm, per objective**, followed by a full-length re-run at the selected LR. Every heat map is published, every grid-edge argmax is reported and the grid extended. Two `α` schedules run throughout: `α/r = 2` fixed (standard) and `α = 32` fixed (which makes the optimal LR approximately rank-independent under `1/r` scaling). **If DeLoRA-PC's advantage vanishes under per-rank LoRA retuning, that is the headline.**

**Prior-art position, stated plainly.** Normalized low-rank adaptation is published (DeLoRA, ICLR 2025, in PEFT). LoRA-in-RL at low rank is published (LoRA Without Regret). Neither has crossed them: nobody has published a rank-resolved, LR-controlled comparison of *per-component* versus *global* adaptation strength under a verifiable-reward objective with a matched SFT control and a per-problem analysis. That intersection is the contribution, and it is narrow by construction. Two checks to run before writing, neither requiring compute: (i) whether any 2025–2026 GRPO/DAPO-with-PEFT report already published a rank sweep in RLVR — if so the main effect is partly scooped and the per-component contrast (H3) survives, which is the part Gate 1 blocks on, so the plan is robust to it; (ii) whether the PEFT-under-noisy-supervision literature has already run the normalized-adapter × label-noise cell.

**Risks.**

- **R1 — H3 fails and the method has no delta.** The most likely single termination point, tested directly at 4B with MDE 1.6. Project 2's Case B (~55% that the SFT gap collapses to `+1…+2.5`) applies here too. Mitigation is honesty: Gate 1(b) stops the program, and the write-up is a rigorous null on per-component scaling.
- **R2 — The `s = 0` initialization trap.** Directions are frozen until scales escape zero; a high dead-group fraction may starve the escape, and the move to strong Instruct models raises the all-correct dead-group fraction (§3.4). Fully instrumented, two pre-registered mitigations, and characterized as a *function of the dead-group fraction* in vision for ~5 GPU-h before any LLM run.
- **R3 — Learning-rate confound.** Addressed above. Residual risk: a 40%-length LR proxy picks the wrong argmax for a full-length run; mitigated by re-running the two best LRs at full length for the two core arms (inside T1.3/T1.4's seed budget as seeds 1–2).
- **R4 — The 1-bit hypothesis wins: everything ties at every rank.** Plausible and publishable, *provided the RFT-SFT twin shows a gap in the same codebase*: "adapter parameterization is objective-insensitive down to `r = 1`" answers the DeLoRA-PC draft's own future-work question with a clean negative, strengthened by matched gradient-SNR and matched-KL evidence for *why*. This is the strongest argument for never cutting the RFT control under budget pressure even though the interaction is no longer primary.
- **R5 — pass@k regression.** A better-conditioned optimizer may collapse diversity faster: higher pass@1, worse pass@16. Pre-registering pass@16 as a co-primary prevents quietly dropping it.
- **R6 — Step-time miss.** The 37.5 s / 43 s step estimates are the optimistic end of the A6000 band and the 4B multiplier is the bottom of the stated 2.5–3× range. A 2× miss on training time costs ~+480 GPU-h and pushes the program to ~1,136 h, over the ceiling. The pre-registered response, **in this order**: (1) cut GRPO length 600 → 400 steps at both tiers (−33% of training cost, ~−160 h) and re-report the length as a design parameter; (2) drop the `r = 8` spot check (−50 h with twins and evals); (3) drop the conditional 9B extension. **Seeds are the last thing cut, and at 3–5 there is nothing left to cut anyway — which is the second reason the analysis, not the seed count, had to carry the power.**
- **R7 — Reward hacking asymmetry.** Both arms share the rollout path and the verifier, so hacking should be symmetric; if it is not, the accuracy comparison is void regardless of significance. Gate 1(d) makes this blocking.
- **R8 — "Why the sphere?"** A delta without a mechanism gets rejected. The gradient-SNR measurement, the reward-vs-KL Pareto, the radial-drift correlation, and the four-point vision dose–response are the answer, and they are what the paper foregrounds regardless of which way the headline number falls.
- **R9 — Ceiling compression on MBPP-Plus.** New with the model upgrade: an 80%+ base pass rate shrinks the addressable gap and inflates the dead-group fraction. Mitigated by the hard-slice training mix and the pre-registered base-fails stratum (§3.4). If the whole-benchmark and stratum results disagree in sign, that disagreement is the reported result, not a selection opportunity.
- **R10 — CRN under-delivers.** The 1.4-point primary MDE assumes `ρ = 0.7`. If `ρ̂ ≈ 0.4`, the primary MDE is 1.9 points and the interaction is 2.2. `ρ̂` is measured and published from the calibration runs *before* unblinding, and the revised MDE is what the gate uses. This is the single assumption in §3.6 most worth a reviewer's attention.

### 3.10 Deferred — what still needs more than one A6000

Everything below fits the *scientific* program but not the A6000 in time or memory. Each has an explicit ask and an explicit unlock condition.

| Deferred item | Blocked by | Ask | Unlocked by |
|---|---|---|---|
| **Long-horizon RLVR**: 2048-token completions, `G=16`, 1500 steps, full LiveCodeBench-v6 window, Qwen3.5-9B, 2 arms × 2 objectives × 5 seeds | **Time** (8× the tokens/step of Tier 2; ~50 A6000-h/run → ~1,000 A6000-h) | **~450 H100-80GB-hours** (≈ $0.9k spot) | Gate 2 passing *and* the Tier 1/2 simple effect being positive at 512 tokens. If the effect exists only at short completions, that is itself the finding and this is not run. |
| **Second model family** (Olmo-3-7B), full Tier 2 replication at 5 seeds | Time | **~120 H100-hours** | Gate 2 passing. The one sanctioned departure from the Qwen3.5-only training rule, and deliberate: a single-family result is the most common reviewer objection to this kind of paper, and cross-family generalization is the whole point of the exception. Cheapest high-value use of external compute. |
| **Qwen3.5-27B dense LoRA RL**, 2 arms × 3 seeds | **Memory** (~56 GB LoRA bf16; does not fit 48 GB) and time | **~250 H100-hours**, or the reserved 2×A100 | Only if the effect is *growing* from 4B → 9B. A flat or shrinking scale curve makes 27B uninformative, and the 2×A100 is reserved for later scale demonstration regardless. |
| **Qwen3.5-35B MoE LoRA RL pilot** | Memory + time (MoE routing makes colocated generation memory-spiky) | **~300 H100-hours** | Strictly after 27B, and only if a reviewer asks whether the result is dense-specific. Not a claim this paper makes. |
| **4B full-FT anchor**, 3 seeds | **Memory** (~64 GB for weights + grads + fp32 Adam; full FT caps ~1.5–3B here) | **~100 H100-hours** | Only if a reviewer demands it. LoRA-vs-full-FT in RL is already answered by LoRA Without Regret and re-litigating it is not this paper's job. |
| **2k-step GSPO stability run** | Time | **~120 H100-hours** | Not gated on this project; an algorithm-robustness check that belongs to whoever writes the follow-up. |

**Notably not deferred:** the entire 4B and 9B GRPO program, the 9B confirmation as a *test* rather than a consistency check, the LLM LR sweeps, and the reward-vs-KL Pareto at 9B. The 3090 version deferred ~2,700–9,000 H100-hours; this envelope brings all of the decisive experiments in-house for 656 A6000-hours and leaves **~450 H100-hours as the only ask with real scientific weight** (long-horizon), plus ~120 for a second family. That is roughly $1.1k of external compute against a program that previously needed $25k, requested *after* the effect has been measured rather than in order to measure it.

**On buying a third scale point:** the tempting use of the ~230 remaining in-house hours after the conditional 9B extension is a larger model, and it is the wrong buy — 27B does not fit, and a third point on a scale curve is not a new claim. Those hours stay unallocated as contingency against R6 and R9, which is what the previous two revisions of this section did not do and paid for twice.


---

## Project 4 — Measuring Representation Geometry with von Mises–Fisher Concentration: Finite-Sample Bias, Matched Nulls, and an Audit of the Isotropy Literature

**Venue: ICLR (main track).**

> ### Compute budget: **≈ 60 GPU-hours planned, ≈ 72 GPU-hours with 20% contingency**, on a single RTX 3090 (24 GB).
> Plus **≈ 412 CPU-core-hours** (embarrassingly parallel, runs on the desktop CPU while the GPU is busy) and **≈ 180 GB of selective checkpoint downloads**.
> Hard ceiling is 1000 GPU-h; target is 100. **This project comes in at roughly 7% of the ceiling and under the target.** Nothing in this plan trains a model with ≥ 1B parameters, and the largest training run is an 85M-parameter transformer.
> Tier 0 go/no-go is **4 GPU-hours**.

### 4.0 Why this project is the natural place to start on one 3090

This is the cheapest project in the program by an order of magnitude, and neither the re-scope nor the model policy touched its scientific content. Three structural facts make it cheap:

1. **Fitting a vMF to a model's input-embedding or unembedding matrix requires no forward pass at all.** The estimand is a property of a weight tensor. You open the checkpoint, read `model.embed_tokens.weight` (or `lm_head.weight`), and estimate. With `safetensors` lazy loading you read the *index* first, identify the single shard containing the embedding tensor, and download only that shard — typically 0.2–3 GB, not the whole checkpoint. The model is never instantiated, never moved to the GPU, and its parameter count is irrelevant. **Llama-3-70B's unembedding matrix (128256 × 8192 bf16 = 2.1 GB) is as cheap to analyse as GPT-2's, and the same is true of Qwen3.5-27B (1.6 GB) and the Qwen3.5-35B MoE (1.2 GB).** The 24 GB VRAM limit does not apply to §4.4(a) at all, because §4.4(a) never touches VRAM.
2. **Null-table generation is CPU-bound vMF sampling, not GPU work.** It is a pure Monte Carlo over `(n, p, κ)` cells with no gradients, no model, and no data. It parallelises perfectly across cores, and it is charged to CPU-core-hours below, not GPU-hours, so it never contends with training. The repo's Householder sampler is what makes it affordable: at p = 4096, fp64, `numpy_hh_inplace` runs at **10,789 vec/s/core** against SciPy's **95 vec/s** (measured, `measurements/benchmark_{numpy,scipy}_dim4096.csv`) — a 113× ratio that at p = 8192 becomes 5,509 vs ~20 vec/s (275×). The full null-table campaign is ~200 core-hours with the repo's sampler and would be ~2.4 × 10⁴ core-hours with SciPy's. That is the difference between "runs overnight on a desktop" and "does not happen".
3. **Only the hidden-state studies need forward passes**, they are inference-only (no gradients, no optimizer state, no activations retained), and they run on models that fit comfortably: GPT-2 124M/355M/774M, Pythia-70m/160m/410m/1.4b/2.8b, SmolLM2-135M/360M, Qwen2.5-0.5B, Qwen3-1.7B, and **Qwen3.5-4B** in bf16 (~8 GB of weights), plus **one** current-generation **Qwen3.5-9B** in NF4 4-bit (≈ 6 GB of weights) as a Tier-2 confirmation that findings at 0.5B survive at p = 4096.

The one genuinely new training cost is the nGPT ladder in §4.4(c). That is **pre-training from scratch, not finetuning**, and it is already correctly small: the re-scope replaced a 500–800 H100-hour 0.5B pretrain with a **three-point ladder of 10M / 30M / 85M-parameter nGPTs plus matched baselines and LR sweeps, at ~29 GPU-hours total.** That is a strictly better experiment: three sizes, multiple seeds, and per-arm learning-rate sweeps beat one flagship checkpoint with n = 1.

**A note on model selection, because this section looks like it violates the program's model policy and does not.** The program trains and finetunes only within the Qwen3.5 family (4B, 9B, 27B dense, 35B MoE; Apache 2.0). This section **trains almost nothing** — its ~40 named models are *analysis subjects*: released checkpoints whose weights or activations are being measured, and the policy governs training, not measurement. For a measurement paper, generational and architectural diversity is not legacy debt, it is the independent variable. An embedding atlas whose rows all come from one family and one generation cannot answer "is this property of the family or of transformers", which is the question the atlas exists to ask; and the audit in §4.4(e) must re-run published claims **on the checkpoints those claims were made about** (ELMo, BERT-base, GPT-2 small, Pythia) or it is not an audit. Pythia is retained for a second, stronger reason: it is the only public suite with a dense **~150-checkpoint training trajectory** at matched data order, and nothing in the Qwen3.5 generation replaces it, so every training-dynamics statement in this section rests on it. A third reason applies to §4.4(d): the published induction-head and retrieval-head taxonomies that the head diagnostic validates against exist for GPT-2 small and the Pythia suite and do not exist for 2026 checkpoints. Current-generation coverage is therefore **added rather than substituted**: the Qwen3.5 family (4B, 9B, 27B dense, 35B MoE) joins the atlas in §4.4(a), and Qwen3.5-4B joins the forward-pass set in §4.4(b)/(d), with Qwen3.5-9B as the Tier-2 p = 4096 confirmation.

### 4.1 The thesis and why it is a paper, not a utility

Every claim in the representation-geometry literature — "contextual embeddings are anisotropic", "normalization makes features more uniform", "the model collapses class variability", "adaptation rotates the representation space" — is made with an instrument that has no units, no null distribution, and no sampling theory. The instruments in current use are: average pairwise cosine and the anisotropy/cosine-cone statistic (Ethayarajh, EMNLP 2019, arXiv:1909.00512), IsoScore (arXiv:2108.07344), participation ratio and effective rank, weight/Gram condition numbers, CKA (arXiv:1905.00414), Procrustes disparity, and 2–3D PCA scatter plots read by eye (nGPT's own geometry diagnostics, arXiv:2410.01131, are exactly this class: norms, condition numbers, dot-product histograms). None of them is comparable across dimension, none reports a confidence interval, and none states what value it takes when nothing is going on.

Directional statistics supplies the missing object. The von Mises–Fisher concentration κ is a parameter of a stated generative model on S^{p−1}; it has an MLE, a Fisher information, a likelihood ratio test, a goodness-of-fit test, and a sampling distribution. It is dimension-aware by construction, because the relevant quantity is κ/p rather than κ. It also gives a continuous, null-calibrated version of NC1 from neural collapse (Papyan, Han & Donoho, PNAS 2020), which is *literally* an intra-class concentration statement, and it remains defined when d < C−1, where the simplex-ETF framing breaks and HUG (arXiv:2303.06484) has to substitute hyperspherical uniformity.

So why is this not a two-hundred-line utility function? Because in the regime modern models occupy — p ∈ [768, 8192], n/p between 1 and 25 — the estimator is *catastrophically and directionally* wrong, and nobody has characterized this. The repo's completed benchmark (Aug 2026, Apple M3 Max + one CUDA device, fp64) establishes three things that constitute the paper's motivation and that we cite as preliminary results:

1. `scipy.special.ive` underflows to exactly zero above order ≈ 511. SciPy's log-Bessel is therefore −inf and Sra-with-SciPy returns **NaN for all p ≥ 1024**; safeguarded Newton silently stalls at ~1e-3 relative error rather than erroring. The compiled CUSF backend holds ~1e-13 everywhere. **Every off-the-shelf κ pipeline is silently wrong or NaN in exactly the dimensional range where LLM representations live.** This is a reportable finding on its own and comes with a SciPy bug report.
2. Statistical error dominates numerical error by ~12 orders of magnitude, and is *identical across every numerically accurate solver*. Median relative error of κ̂ against the generating κ at n = 10⁴: **39.8** at (p=4096, κ=1), **9.3** at (p=1024, κ=1), **0.29** at (p=4096, κ=50). Choosing a better solver buys nothing; the entire problem is finite-sample.
3. The finite-sample bias is **systematically upward** and grows as concentration falls. Anyone fitting vMF to embeddings wants to claim "more concentrated than uniform" — and the artifact points the same way as the claim. This is the paper's teeth.

Point (3) is the reason a matched null is mandatory rather than nice-to-have, and the reason an audit of existing practice is warranted. We state plainly in the introduction what is *not* novel: the estimator (Banerjee et al., JMLR 2005; Sra 2012, arXiv:1605.00316), the sampler (Ulrich 1984 + Householder), the second-moment resultant-length correction, and the Cramér–Rao machinery are all textbook directional statistics (Mardia & Jupp; Fisher, Lewis & Embleton). The novelty is (i) the regime, (ii) the demonstration that current ML practice is invalid in it, (iii) the shipped calibration, and (iv) the findings in §4.4. A reviewer who discovers the "this is known statistics" objection unaided will reject; stated first by us, it is a scope claim.

**One global design rule, stated once and enforced everywhere.** n = 1 gives R̄ = 1 exactly and κ̂ = R̄(p − R̄²)/(1 − R̄²) is a division by zero. Therefore **no per-input statistic in this program is ever an estimated κ.** Per-input quantities are vMF *log-likelihoods* under a κ fitted on a pooled set, or raw cosines. Any figure with one point per token/example/document plots a likelihood, never a κ̂.

### 4.2 Contribution 1: the estimator's behaviour in the LLM regime

**The (n, p, κ) bias surface — re-scoped grid.** p ∈ {64, 128, 256, 512, 768, 1024, 1536, 2048, 4096, 8192} (10 values, chosen to cover every model width in §4.4(a)); κ/p on a log grid from 0.001 to 2, **9 points** (was 12); n ∈ {10², 10^2.5, …, 10⁶}, 9 points. Replicates: **1000** for n ≤ 10⁴, **200** at 10^4.5 and 10⁵, **50** at 10^5.5, **25** at 10⁶ (was 1000/200/50 across the board). Total sampled vectors ≈ **6 × 10⁹**, down from ~10¹⁰. What this costs: at the measured `numpy_hh_inplace` fp64 throughputs (32,808 vec/s at p=1024; 10,789 at p=4096; 5,509 at p=8192), the p-weighted average is ~15k vec/s/core, giving **≈ 110 core-hours of sampling** plus ~60 core-hours of estimator inversion and bookkeeping — call it **200 CPU-core-hours** with scheduling slack, ≈ 25 h wall-clock on an 8-core desktop, 13 h on 16 cores. A GPU fallback exists (`torch_hh_cuda` at 584k vec/s fp64, p=4096) and would finish in ~4 GPU-h, but we keep this on the CPU on purpose so it never blocks the GPU. Note that the p grid is unchanged by the addition of the Qwen3.5 checkpoints: their widths (2560 / 4096 / 4096 / 5120) already fall inside it, so the atlas grows without enlarging the null-table campaign.

For each cell we report median relative bias, RMSE, interquartile range, and the empirical distribution of κ̂ — not just the median, because the tail is what produces false discoveries. The preliminary run already establishes the qualitative shape: **κ/p governs everything, not p**; above κ/p ≈ 0.2, n = 10–25p buys 1% accuracy; below κ/p ≈ 0.05, even n = 10⁵ does not. The full study adds: the surface at fixed resolution across all three axes, the bias *direction* map, the corrected estimator, and the Cramér–Rao floor that explains where correction cannot help. The cut in replicates costs precision on the *tail* quantiles at n ≥ 10^5.5 only; the median-bias surface, which is the load-bearing object, is unaffected.

**Why the bias points the wrong way, in closed form.** For n uniform unit vectors in R^p, E‖x̄‖² = 1/n exactly, so R̄ ≈ n^{−1/2} to leading order, essentially independent of p. Pushing that through Banerjee's closed form κ₀ = R̄(p − R̄²)/(1 − R̄²) gives a null expectation of roughly **κ̂_null ≈ p/√n**. At p = 4096 with a V = 32768 vocabulary this predicts κ̂ ≈ 22.6 *from pure noise*; at p = 768 with V = 50257 it predicts ≈ 3.4. We present this as an analytic prediction to be verified numerically on the grid, and it is the single most useful number a practitioner can carry away: **if your reported κ̂ is not several times p/√n, you have measured your sample size.**

**A bias-corrected estimator that takes n as an argument.** Two stacked corrections:

- *Exact second-moment correction.* With ρ = A_p(κ) = I_{p/2}(κ)/I_{p/2−1}(κ), E[x_i·x_j] = ρ² for i ≠ j, so E[R̄²] = 1/n + (1 − 1/n)ρ² exactly. Hence ρ̂²_bc = max(0, (nR̄² − 1)/(n − 1)) is unbiased for ρ², and R̄_bc = √ρ̂²_bc inverted through A_p gives κ̂_bc. This is exactly zero in expectation under the uniform null — it removes the entire leading-order artifact — and it costs one extra line. It is standard directional statistics; as far as the brief's survey goes it has never been applied in the ML measurement setting.
- *Median-unbiased parametric calibration.* Residual bias enters through Jensen on the nonlinear inversion A_p^{−1}. Since A_p^{−1} is monotone, we calibrate: simulate n draws at candidate κ, measure the median κ̂_bc, and solve for the κ whose simulated median matches the observed value. Monotone, one-dimensional, and *only affordable because of the sampler in §4.3* — each calibration is O(10²) inner simulations at the user's own (n, p), i.e. a sub-second call at p = 1024 and a few seconds at p = 8192.

**A third, cheaper estimator for the known-mean case, which turns out to matter.** When the mean direction μ is known a priori rather than estimated — which is exactly the situation in §4.4(c), where μ is the (unit) hidden state and the sample is the emitted token's unembedding row — the sufficient statistic is R̄_known = (1/n) Σ x_i·μ_i, and κ̂ solves A_p(κ) = R̄_known directly. **This estimator has no leading-order upward bias**, because the p/√n artifact comes entirely from fitting μ to the same data. Under the uniform null, E[R̄_known] = 0 and sd(R̄_known) = 1/√(np), so κ̂_null ≈ p/√(np) = √(p/n) — at p = 512, n = 10⁷ that is **0.007**, versus 0.16 for the estimated-mean version. Distinguishing the two cases is a small technical contribution that materially changes which experiments are feasible, and it is why the corrected nGPT test in §4.4(c) is well-conditioned.

**Confidence intervals and a refusal mode.** Parametric-bootstrap CIs from the same machinery, with empirical coverage of the nominal 95% interval reported across the grid as a table (a coverage table, not a claim). Cost: **~120 CPU-core-hours** on a reduced grid (p ∈ {128, 768, 1024, 4096, 8192} × 6 κ/p × 6 n × 500 bootstrap resamples). The API carries a **feasibility flag**: when the estimated κ/p falls below the resolvable threshold at the user's n, the function returns "not resolvable at this n" rather than a number. Refusing to emit a number is a feature; it is what would have prevented several of the claims we audit in §4.4(e).

**A Cramér–Rao floor, so the infeasibility map is a theorem and not an anecdote.** Fisher information for κ from n i.i.d. vMF samples is n·A′_p(κ) with A′_p(κ) = 1 − A_p(κ)² − (p−1)A_p(κ)/κ. The resulting sd(κ̂) ≥ (n A′_p(κ))^{−1/2} converts the empirical "n = 10⁵ is not enough below κ/p = 0.05" into a statement about *any* estimator. This upgrades the feasibility map from a benchmark artifact into a bound, and it is nearly free — **~10 CPU-core-hours**, computed with the same CUSF log-Bessel backend.

**Shipped nulls, plural.** Null quantiles (50/90/95/99/99.9) of R̄ and κ̂ over the (n, p) grid, plus a fitted interpolant, for three nulls: (a) uniform on S^{p−1}; (b) mean-removed / common-direction-removed embeddings, matching Mu & Viswanath's all-but-the-top preprocessing; (c) a **Gaussian null with the empirical covariance spectrum**, which is the one that matters for the audit — it separates "the covariance is anisotropic" from "the directions are concentrated", two claims the literature routinely conflates. Null (c) is per-checkpoint and is generated on demand from the checkpoint's own embedding covariance; at V ≈ 1.5 × 10⁵ rows and 200 replicates it is ~3 × 10⁷ vectors per checkpoint, i.e. seconds even at p = 5120.

**Head-to-head against the incumbent instruments.** A synthetic benchmark with known ground truth: mixtures of vMF at known κ, at p ∈ {128, 768, 4096}, subjected to named nuisance transformations (global rescaling, injection of a common direction, addition of a rank-k component, per-coordinate scaling). Metrics, defined: (i) AUC of each statistic in discriminating κ = 0 from κ = κ₁ at fixed n; (ii) false-positive rate at nominal α = 0.05; (iii) rank correlation between the statistic and true κ *across different p* (the dimension-comparability test that all incumbents should fail). Baselines: average pairwise cosine, Ethayarajh's anisotropy, IsoScore, participation ratio, effective rank, condition number. **~20 CPU-core-hours.** **The ablation that kills the claim:** if IsoScore, given the same matched null, matches κ̂_bc on AUC and calibration at every p, then the instrument contribution collapses to "we gave IsoScore a null" — still worth writing, but not a contribution bullet.

**Deliverable:** a pip-installable `vmf-measure` wrapping the repo's `src/vmf_kappa.py`, `src/vmf_kappa_torch.py`, `src/cusf_cpu.py` and `src/torch_log_iv.py`, exposing `kappa_hat(X, mu=None, correct=True, null="uniform", ci=0.95)` → (κ̂_bc, CI, null quantile, p-value, feasibility flag), defaulting to the CUSF/torch backend because SciPy is unusable at p ≥ 1024.

**Likely negative result and its value:** the correction may trade bias for variance and fail to improve RMSE where the naive estimator fails worst. We pre-commit to reporting RMSE, not bias alone. If correction does not help below κ/p ≈ 0.05, the finding becomes "no estimator can resolve this regime, here is the CRLB proving it, and therefore the isotropy question is *not answerable* at vocabulary scale for p ≥ 4096" — a citable negative result that changes what people are allowed to claim.

### 4.3 Contribution 2: the fast sampler as a precondition, not a headline

The repo's samplers beat the SciPy reference by **113× at p = 4096 and ~275× at p = 8192** on CPU fp64 (measured; `measurements/`), and the CUDA Householder path reaches 584k vec/s at p = 4096 fp64 and 1.42M vec/s at p = 1024. This belongs in the paper as one paragraph plus a table in the experimental-setup section and a released artifact — **not as a contribution bullet.** The algorithm is Ulrich (1984) with a Householder reflection, the same as the hyperspherical VAE's; what is new is engineering: a single reflection mapping e₁ → μ, a fused `addmm` rank-one update, full batching, dtype-generic fp16/bf16/fp32/fp64, device-native execution, and closed forms at d = 2, 3. An ICLR reviewer will not accept a speedup as a scientific claim, and pretending otherwise invites a rejection.

The correct framing is necessity. The null tables and the median-unbiased calibration of §4.2 require ~**6 × 10⁹** sampled unit vectors at dimensions up to 8192. At the repo's throughput that is ~200 core-hours: one desktop, overnight, twice. At the SciPy reference throughput it is ~2.4 × 10⁴ core-hours: a cluster allocation this project does not have. **The speedup is precisely the difference between a single-3090 researcher being able to produce the calibration and not.** That is a stronger and more honest framing than "127× fast", and it is only available at this budget. The same argument applies at *call* time: every user re-runs a bootstrap at their own (n, p) inside an analysis loop.

The scientifically load-bearing part is **validation of the sampler**, since the null tables inherit its correctness. We report goodness-of-fit of the marginal of t = x·μ against its known density ∝ (1 − t²)^{(p−3)/2} e^{κt} (KS and Anderson–Darling, across the full (p, κ) grid, all dtypes), plus rejection-acceptance rates as a function of p — the hyperspherical VAE reports 1.0–1.46 and *improving* with dimension, which we confirm or refute at p up to 8192. A rejection sampler whose efficiency does not degrade with p is what makes the whole calibration tractable, and that fact is worth a figure. **Cost: 1 GPU-h** (this is the one sampler item charged to the GPU, because we must re-measure the CUDA throughput table on the actual 3090 rather than reusing the recorded CUDA rows from unknown hardware — the published table must state the device it was measured on).

### 4.4 Contribution 3: applications that produce actual findings

Inference-only except (c) and (f). Critical engineering note that makes several of these cheap: **κ̂ requires only the resultant vector Σx and the count n.** Activations never have to be stored — stream and accumulate in fp64 on the host. Per-document partial resultants (for the block bootstrap) are p floats per layer per document; at p = 1024, 32 layers, 10⁴ documents that is 2.6 GB, not the 40 TB of raw activations.

#### (a) The embedding atlas — zero GPU-hours, ~31 checkpoints plus a 150-point training trajectory

Input-embedding and unembedding matrices, read straight out of the safetensors shard, model never instantiated. **This is where "many small models beat one big run" pays off most: ~31 checkpoints for the price of the download.** Cost: **~42 CPU-core-hours + ~140 GB of selective shard downloads** (~95 GB for the atlas rows, ~45 GB for the Pythia trajectory; shard granularity, not tensor granularity, is what sets the number).

**Why this table deliberately spans four generations of models.** The atlas is the one place in the program where *breadth of provenance is the measurement*. The organising question — does directional concentration track width, vocabulary size, tokenizer, tied-vs-untied embeddings, or training recipe — is unanswerable inside a single family, and the two extreme rows (Mistral-7B at V/d = 8, Gemma-3-1B at V/d = 228) are the ones that define the feasibility axis. Older checkpoints are therefore retained on purpose, not by inertia. Current-generation coverage is **added**: the Qwen3.5 family enters as the four bolded entries below, and because a vMF fit needs only the embedding tensor pulled from one safetensors shard, **the 27B dense and 35B MoE cost the same as a 1B model** — 1.2–1.6 GB of download, no VRAM, no forward pass, no relevance of the 9B's hybrid layer stack to the estimand. There is no budgetary reason to exclude them and no scientific reason to exclude their predecessors.

For each: κ̂_bc with CI and null quantile, raw / mean-centered / top-k-PC-removed; movMF mixtures with BIC selection on held-out rows; a Rayleigh test — pre-empting the "your instrument assumes a distribution the data doesn't have" objection.

The organising quantity is **V/d** (= n/p) and the predicted noise floor **κ̂_null ≈ d/√V**. Ranked, this reframes the atlas: large-vocabulary small models are the *statistically privileged* checkpoints, not the big ones.

| Checkpoint | d (p) | V (n) | V/d | κ̂_null ≈ d/√V | Loadable here? |
|---|---|---|---|---|---|
| Gemma-3-1B | 1152 | 262144 | 228 | 2.2 | Tensor only (gated repo; accept license) |
| Qwen2.5-0.5B / Qwen3-0.6B | 896 / 1024 | 151936 | 170 / 148 | 2.3 / 2.6 | Tensor + full bf16 forward |
| Gemma-3-4B | 2560 | 262144 | 102 | 5.0 | Tensor + bf16 forward (8 GB weights) |
| Pythia-70m | 512 | 50304 | 98 | 2.3 | Tensor + forward |
| SmolLM2-135M / 360M | 576 / 960 | 49152 | 85 / 51 | 2.6 / 4.3 | Tensor + forward |
| GPT-2 124M / Pythia-160m | 768 | 50257 / 50304 | 65 | 3.4 | Tensor + forward |
| CLIP ViT-L/14 (text tower) | 768 | 49408 | 64 | 3.5 | Tensor + forward |
| **Qwen3.5-4B** | 2560 | 151936 | 59 | 6.6 | Tensor + bf16 forward (8 GB weights) |
| GPT-2 355M / Pythia-410m | 1024 | ~50k | 49 | 4.6 | Tensor + forward |
| GPT-2 774M / 1.5B | 1280 / 1600 | 50257 | 39 / 31 | 5.7 / 7.1 | Tensor + forward (774M yes, 1.5B bf16 forward yes at short ctx) |
| Qwen3-8B | 4096 | 151936 | 37 | 10.5 | **Tensor only** (forward needs NF4) |
| **Qwen3.5-9B** | 4096 | 151936 | 37 | 10.5 | **Tensor** + NF4 forward in Tier 2 (~6 GB) |
| **Qwen3.5-35B (MoE)** | 4096 | 151936 | 37 | 10.5 | **Tensor only** — 1.2 GB shard, forward is out of budget |
| Llama-3.1-8B | 4096 | 128256 | 31 | 11.4 | **Tensor only** (forward needs NF4; gated) |
| **Qwen3.5-27B** | 5120 | 151936 | 30 | 13.1 | **Tensor only** — 1.6 GB shard, trivially loadable |
| Pythia-1.4b / 2.8b | 2048 / 2560 | 50304 | 25 / 20 | 9.1 / 11.4 | Tensor + bf16 forward |
| OLMo-2-7B / 13B | 4096 / 5120 | 100352 | 25 / 20 | 12.9 / 16.2 | **Tensor only** |
| Llama-3-70B | 8192 | 128256 | 16 | 22.9 | **Tensor only** — 2.1 GB shard, trivially loadable |
| Pythia-6.9b / 12b | 4096 / 5120 | 50304 | 12 / 10 | 18.3 / 22.8 | **Tensor only** |
| Mistral-7B | 4096 | 32768 | 8 | 22.6 | **Tensor only** — worst case in the atlas |

That is 28 language-model entries; DINOv2 and the CLIP/SigLIP image towers bring the atlas to ~31. The Qwen3.5 rows use nominal (d, V); the loader reads both from `config.json` and recomputes V/d and κ̂_null per checkpoint, so a differing hidden size or vocabulary changes the row, not the analysis. Tied embeddings (where `lm_head` is `embed_tokens`) are flagged and counted once.

**The Pythia training trajectory — the one thing no current-generation suite can replace.** Pythia publishes ~150 intermediate checkpoints per model size at matched data order. Because a vMF fit is a weight-tensor operation, the entire trajectory of Pythia-160m and Pythia-410m (embedding-bearing shard only, ~0.15 GB per revision, ~45 GB across ~300 revisions) gives **κ̂_bc(t) over training** at zero GPU cost: when does the unembedding geometry stop moving, does κ̂ rise monotonically or overshoot, and does it track the loss curve or the LR schedule. No Qwen3.5, Llama or Gemma release exposes anything comparable — intermediate checkpoints are simply not published — so this sub-study is retained on Pythia specifically and is labelled as such in the paper. **Cost: ~8 CPU-core-hours, included in the ~42 above.** The same trajectory supplies the *only* within-run null calibration in the section: 150 correlated-but-ordered fits let us show that the null quantile is flat in t while κ̂ is not.

DINOv2 and the CLIP/SigLIP image towers have no token-embedding matrix; for those the analogous objects are the patch-embedding filter set (n = d_model rows in R^{3·14·14 = 588}) and any prototype/classifier head, both of which we report separately and flag as a different estimand.

**Expected outcome is partly negative and we say so up front:** the n-axis is comfortable (V/d ≥ 25) for roughly half the atlas, but n/p ≥ 25 only buys 1% accuracy *when κ/p > 0.2*; below κ/p ≈ 0.05 the CRLB bites regardless of V. So the predicted split is: Gemma/Qwen-class models (including Qwen3.5-4B at V/d = 59) resolvable, Mistral/Pythia-large not, and the interesting result is whichever side of the line real embedding matrices actually fall on. That is the motivating figure of the paper, not its finding.

#### (b) Layerwise κ on activations — small models, breadth over scale

Eleven models, all bf16, forward-only, no gradients: GPT-2 124M/355M/774M, Pythia-70m/160m/410m/1.4b, SmolLM2-360M, Qwen2.5-0.5B, Qwen3-1.7B, **Qwen3.5-4B**. Three-plus families and four generations, on purpose: the cross-family question below is the finding, and it dies if the set is homogenised. **10M tokens each** of C4 / Pile-validation, packed into 1024-token sequences, streamed resultants. Per layer, per token-position bucket, per probing class.

Cost: forward FLOPs ≈ 2ND, so 10M tokens through a 410M model is 8 × 10¹⁵ FLOPs ≈ 400 s at a realistic 20 TFLOPS bf16 on a 3090; the 1.7B model is ~30 min; **Qwen3.5-4B is 8 × 10¹⁶ FLOPs ≈ 1.1 h** and is the single largest item here. The eleven models sum to ≈ 2 × 10¹⁷ FLOPs ≈ 2.8 h of pure compute; with forward hooks, fp64 host-side accumulation and data I/O the realised factor is ~2.3×, so **6.5 GPU-h** (was 5.0 for ten models).

Here n = 10⁷ so the null is small (κ̂_null ≈ p/√n ≈ 0.65 at p = 2048) and almost everything is resolvable. **The trap, which we handle explicitly:** tokens within a document are not independent, so the i.i.d. null is wrong and the effective n is far smaller — the null must be a **document-level block bootstrap**, and we quantify how much the i.i.d. null over-rejects. Substantively, this re-measures Ethayarajh's "anisotropy grows with depth" with a proper unit and a proper null, tests whether late-layer concentration is genuine directional concentration or a rank-one common-direction artifact (remove top PC, re-measure), and — because we have eleven models spanning 70M to 4B, four families and four generations — asks whether the depth profile is *architecture-family-specific* and whether it has *changed across generations*, which no single-model and no single-family study can ask. Tier 2 adds Qwen3.5-9B in NF4 to check that the profile shape survives to p = 4096.

#### (c) The nGPT s_z test — corrected estimand, and now affordable

**The original formulation of this test was wrong and we fix it.** nGPT's learned logit scale s_z ≈ 60.8 was to be compared against "κ̂ of the unembedding rows". That is the wrong estimand: nGPT's unembedding rows are unit-normalised and there is no mechanism concentrating them around a common direction — they are near-uniform on S^{p−1} by design, so their κ̂ is a near-vacuous quantity dominated by the p/√V noise floor, and agreement with s_z would be a coincidence rather than a confirmation.

**The correct estimand.** nGPT's output distribution over the vocabulary is p(y | h) ∝ exp(s_z · ĥ·ê_y) with both ĥ and ê_y unit vectors. That is *exactly* a vMF density with mean direction ĥ and concentration s_z, evaluated on the discrete support {ê_y}. So the testable claim is about the **angular residuals of the emitted token's unembedding row around the hidden state**:

> For each token position t, take the unit hidden state ĥ_t and the unit unembedding row ê_{y_t} of the token actually emitted (we run both variants: y_t = ground-truth next token, and y_t = argmax sample). Transport ê_{y_t} by the rotation carrying ĥ_t → e₁, giving a residual direction r_t ∈ S^{p−1} with known mean direction e₁. Pool r_t over ~10⁷ token positions and estimate κ with the **known-mean** estimator of §4.2, i.e. solve A_p(κ) = (1/n)Σ ĥ_t·ê_{y_t}. **Pre-registered hypothesis: κ̂_res / s_z ∈ [0.9, 1.1].**

Statistical conditioning, which is now excellent: n = 10⁷, p = 512 gives n/p ≈ 2 × 10⁴, and with a known mean the null is κ̂_null ≈ √(p/n) ≈ **0.007**. The CRLB gives sd(κ̂) ≈ √(p/n) ≈ 0.007 at κ ≈ 60 — a relative precision of 10⁻⁴. **This is so precise that a null-hypothesis significance test is meaningless: any discrepancy will be "significant".** We therefore pre-register a *tolerance* (the ±10% band above), not a p-value, and report the ratio κ̂_res/s_z with its trend across model sizes. Stating this in advance is what stops a reviewer from reading a 4σ deviation as a refutation of a claim that was only ever approximate.

**Which checkpoint?** NVIDIA did not release nGPT weights; only training code and community reimplementations (`NVIDIA/ngpt`, `lucidrains/nGPT-pytorch`) exist. So this test requires training. This is the **only** language-model training in the section, it is **pre-training from scratch** rather than finetuning a released base model, and it therefore sits outside the family policy by construction: there is no Qwen3.5 nGPT to start from, the policy governs finetuning of released bases, and the object under test is the architecture itself. The original plan's answer — a 0.5B nGPT at p = 1024 for 500–800 H100-hours — is 1500–2400 3090-hours and is out of budget by more than an order of magnitude. **The re-scope: a three-point ladder of tiny nGPTs trained from scratch, which is a better experiment anyway**, because s_z is a learned per-model scalar and the hypothesis is a *correspondence* that should hold at every scale, so three sizes with seeds test it three times and reveal any trend in p. The ladder is unchanged from the re-scoped plan — it is already correctly small, and the model policy does not touch it.

| Config | d (p) | layers | non-emb params | tokens (FineWeb-Edu, ctx 512) | arm | GPU-h |
|---|---|---|---|---|---|---|
| A | 256 | 6 | ~10M | 200M | nGPT × 3 seeds | 0.8 |
| A-base | 256 | 6 | ~10M | 200M | pre-LN GPT × 3 seeds | 0.5 |
| B | 512 | 8 | ~30M | 600M | nGPT × 2 seeds | 4.8 |
| B-base | 512 | 8 | ~30M | 600M | pre-LN GPT × 2 seeds | 3.0 |
| LR sweeps (10% token horizon, 5 LRs × 2 arms × 2 sizes) | — | — | — | — | both | 3.0 |
| **C (Tier 2 only)** | 768 | 12 | ~85M | 800M | nGPT × 1 seed | 9.0 |
| **C-base (Tier 2 only)** | 768 | 12 | ~85M | 800M | pre-LN GPT × 1 seed | 5.7 |
| **C LR sweep (Tier 2)** | 768 | 12 | — | 10% horizon | both | 2.3 |

(nGPT is priced at 1.6× the baseline's per-token cost for its normalization and scaling overhead. Baseline FLOPs from 6ND at 20 TFLOPS effective bf16.)

**Ladder roll-up, stated explicitly because the tiers double-count otherwise.** The whole ladder is **29.1 GPU-h**: rows A…LR-sweeps sum to 12.1 and rows C…C-LR sum to 17.0. Of the 12.1, **1.5 is pre-paid in Tier 0** as item T0.5 (config-A single seed plus the p=256 LR sweep), so **Tier 1 carries 10.6** and **Tier 2 carries 17.0**.

**The effective-learning-rate confound applies here and the LR sweeps are non-negotiable.** nGPT is a *normalized-weight* architecture and the baseline is not, so the effective step size scales like η/‖w‖² and every nGPT-vs-GPT comparison is silently also a learning-rate comparison. If we ran one LR per arm and nGPT's s_z came out different, we could not tell an architecture effect from a step-size effect. **At this budget there is no excuse for skipping the sweep: five learning rates at a 10% token horizon on the p=256 model costs ~0.4 GPU-h total across both arms.** This is the compensating advantage of small scale — the ablation that a flagship-run paper cannot afford is nearly free here, and we say so in the paper. Sweep η ∈ {3e-4, 1e-3, 3e-3, 1e-2, 3e-2} per arm, pick by 10%-horizon validation loss, then run the full budget at the per-arm winner.

Interpretation is informative either way: agreement is the first quantitative confirmation that nGPT's learned scale *is* the concentration of its own output distribution, closing a loop nGPT left open; disagreement (and especially a systematic drift of κ̂_res/s_z with p) says the scale is an optimizer-side artifact of rotational equilibrium (Kosson, arXiv:2305.17212) rather than a geometry statement. The matched baselines give the control: in a pre-LN GPT the analogous quantity is ‖W_U‖-induced effective temperature, and we report κ̂_res against it to show the correspondence is specific to the normalized architecture and not a generic property of trained transformers.

#### (d) The attention-head diagnostic — the best statistical regime in the paper

Per-head query/key/output vectors at head dimension p_h ∈ {64, 128}, over 10⁶ tokens: **n/p ≈ 1.6 × 10⁴, κ̂_null ≈ p/√n = 0.064** — by far the most comfortable regime here, and it is *unaffected* by the move to small models, because head dimension is 64 in GPT-2 small, in Llama-3-8B and in Qwen3.5-4B alike. **This is the single most budget-efficient experiment in the project and neither the re-scope nor the model policy weakened it at all.**

Rides on the same forward passes as (b) with extra q/k/o hooks: **+3.5 GPU-h** across the eleven models. Questions: does κ̂ of a head's key distribution predict its attention entropy? Does it separate induction heads (prefix-matching score, Olsson et al. 2022) from retrieval heads (arXiv:2404.15574) from attention-sink heads (StreamingLLM, arXiv:2309.17453)? Does κ̂ of per-head output directions track ablation importance? GPT-2 small and Pythia-160m+ have well-documented induction heads, so validation targets exist at exactly the scales we can afford — this is the sharpest concrete reason the older checkpoints stay in the set: they are the only ones with published head taxonomies to validate against, and a diagnostic with no labelled positives is not a diagnostic. GQA models (Qwen2.5, Qwen3, Qwen3.5) add a free wrinkle: KV heads are shared across query heads, so κ̂ of the shared key distribution versus the per-query-head distributions is a structural comparison MHA models cannot offer.

**Power, recomputed for the enlarged set.** Eleven models give ~1800 heads (was ~1500; this is a deliberately conservative count, since only heads with a computable prefix-matching score enter the ROC). Taking induction-head prevalence at ~10%, that is ~180 positives against ~1620 negatives; by Hanley–McNeil the standard error of a pooled AUC of 0.70 is ≈ 0.022, so the pooled AUC ≥ 0.75 Tier-2 criterion is separated from the 0.70 Tier-1 threshold by more than 2 SE and from chance by ~9 SE. The binding constraint is the *per-model* CI, not the pooled one: the smallest model (Pythia-70m, ~48–96 heads, ~5–10 positives) has SE ≈ 0.09–0.13 and can only exclude 0.5 if its AUC exceeds ~0.68–0.76, which is why the gates in §4.7 are stated pooled with per-model CIs as a consistency check rather than per-model thresholds.

#### (e) The audit of the isotropy literature

Re-run the headline measurements of Ethayarajh (2019), Mu & Viswanath's all-but-the-top, and IsoScore's published model rankings, at *the n each paper actually used*, against matched nulls, with the bias direction accounted for. Rides on the same activation streams as (b): **+2.5 GPU-h**, mostly for the reduced-n subsampling replicates and the document-level block bootstrap. Deliverable: a table of published claims × (original statistic, matched-null value, corrected statistic, survives / does not survive / not resolvable).

Helpfully, the audit's targets are *themselves* small-model studies — Ethayarajh's headline results are on ELMo, BERT-base and GPT-2 small — so re-running them at their original scale is native to this budget rather than a compromise. **This is the hardest constraint on model choice in the section: an audit that substituted current-generation checkpoints for the ones the claims were made on would not be an audit, it would be a different experiment.** The published checkpoints are the data, and the model policy does not reach them because nothing here is trained. **Calibrate the ambition honestly:** at n = 10⁶ tokens, contextual anisotropy is far above any null and will survive; what is at risk are the small-n claims — per-word-type analyses, per-layer-per-class subsets, few-shot prototype studies, and any result computed on a few thousand vectors at p ≥ 1024, where κ̂_null ≈ p/√n is comparable to the reported effect. The publishable form is "most of the literature holds; here are the k claims that are sampling noise; here is the one-line rule (compare to p/√n) for knowing which is which." Claiming to demolish the field would be overreach and would not survive review. Qwen3.5-4B enters this table only as a *forward-looking* row — "does the 2019 anisotropy claim still describe a 2026 model" — clearly separated from the reproduction rows.

#### (f) Controlled vision validation — the only setting with ground-truth class structure

Everything above measures models whose true geometry is unknown. Small-scale vision supplies a setting where the class structure *is* the ground truth and n per class is a knob: within-class κ̂ on penultimate features is a continuous, null-calibrated NC1.

CIFAR-100 (n = 500/class, p = 512 for ResNet-18) sits at n/p ≈ 1 and **Tiny-ImageNet** (200 classes, 64×64, n = 500/class, p = 512) at the same — precisely on the feasibility wall, which makes them the ideal demonstration that the wall is real and where the corrected estimator earns its keep. We sweep n per class by subsampling from 25 to 500 and show the κ̂-vs-n curve converging (or not) to the value obtained on the full training set, against the matched null at each n. STL-10 (p = 512, only 500 labelled/class) and CIFAR-10 (n = 5000/class, n/p ≈ 10) bracket the regime from both sides.

Cost: reuse Project 1's and §6.5's CIFAR-100 / Tiny-ImageNet ResNet-18 checkpoints at zero marginal cost where they exist; otherwise **3 GPU-h** trains 2 seeds of ResNet-18 on CIFAR-100 (~1.5 GPU-h each, 200 epochs, mixed precision) on the 3090 as a fallback — this is exactly the "small vision work" the hardware envelope assigns to the 3090, and the ResNet is trained from scratch, so no family substitution applies. Feature extraction is minutes. This is the cheapest credibility in the paper: a figure showing κ̂ and its null on data where we know the answer.

### 4.5 Relationship to the other projects

This project ships one library, one null-table artifact, and one feasibility calculator that Projects 1–3 all call, so that geometry numbers are comparable *across the program's papers* — same estimator, same correction, same null, same CI convention. The instrument is model-agnostic by construction: it consumes unit vectors, so the sibling projects' move to Qwen3.5 changes their compute, not this project's interface.

- **Project 1 (continual learning, nMLP).** Converts the PCA eyeball — "the nMLP's embeddings move less during phase 2" — into a measurement: for each sample, the unit angular displacement of its normalized representation between phase-1-end and phase-2-end; fit vMF to the displacement directions and report κ̂_bc against a matched null at the same (n, p). Also supplies **κ̂ as a continuous NC1**, which stays defined at d < C−1. And it supplies the guardrail: at p = 512 with n = 1000 per class, κ̂_null ≈ p/√n ≈ 16, so any within-class κ below that is noise. Project 1 reports κ̂ *alongside* CKA, Procrustes disparity, and subspace principal angles, never instead of them.
- **Project 2 (DoLoRA / normalized adapters).** DoLoRA is largely a rediscovery of DeLoRA (arXiv:2503.18225, ICLR 2025, already merged into HuggingFace PEFT), which normalizes the low-rank update and learns a per-layer scalar boundary; the remaining novel delta is **per-component scales s_i versus a single global scalar**, and DoLoRA's own Appendix A proves a global scale is a strictly worse approximator when the target's singular values are non-uniform. The sharp question — "does per-component strength beat a single global scale?" — is architecture-agnostic and needs no large model; it is testable on ViT-B/16 over VTAB-1k on the 3090. Where Project 2 wants a language-model arm, the policy applies and the arm is **Qwen3.5-9B LoRA on the A6000** (~20–24 GB bf16 with the base frozen, so no base gradients or optimizer states); **Qwen3.5-27B and the 35B MoE need ~56 GB for LoRA, do not fit a single A6000, and are out of scope for that arm** (they would need the reserved 2×A100). This project's service to Project 2 is (i) that sharpening and (ii) **a prohibition**: at r ∈ {2,…,32} and p = 4096, n = r is hopeless — the feasibility map says nothing about a rank-8 factor set is resolvable, so Project 2 must not report κ̂ over factor directions. The defensible target is κ̂ of the ΔW-induced representation shift over the eval set, where n is large. Project 2 inherits the same LR-sweep mandate for the same reason as §4.4(c): normalized adapters change the effective step size, so DoLoRA-vs-LoRA is a step-size comparison unless each arm gets its own sweep — and on VTAB-1k a full 19-task sweep is 1–3 GPU-h, so there is no excuse.
- **Project 3 (RL on code generation).** κ̂ of the policy's hidden-state direction distribution across RL steps, as a diversity/collapse diagnostic comparable across model sizes in a way token-level entropy is not. Under the model policy the policy is now **Qwen3.5-4B** (the family's smallest member) rather than a 0.5B-class model, which raises Project 3's per-token cost by roughly 2.5–3× but changes nothing here: p rises from ~896 to 2560, n is still in the millions of generated tokens, so κ̂_null ≈ p/√n ≈ 2560/√10⁷ ≈ 0.8 and feasibility stays comfortable. Requires the block bootstrap from §4.4(b) since completions are correlated within a prompt. If Project 3 moves to Qwen3.5-9B LoRA on the A6000, the reference policy is free via adapter-disable (~18 GB saved) and the diagnostic is unchanged at p = 4096 (κ̂_null ≈ 1.3).

### 4.6 What would make this fail as a paper

**The dominant risk: a measurement paper with no surprising finding is a workshop paper.** ICLR rewards "we found X", not "we built a better ruler". The bias surface, the corrected estimator, the CRLB and the null tables are Section 3 material; they are necessary and they will not carry acceptance alone.

Secondary risks, ranked: (2) *"This is known statistics."* Mitigated only by stating the scope claim in the first paragraph (§4.1). (3) *The audit has no teeth* — everything published survives, and the paper reduces to a tool report. (4) *Unimodality* — if real embedding distributions are strongly multimodal, a single κ is the wrong summary and the instrument measures nothing; mitigated by the movMF/BIC diagnostic and by being willing to publish "vMF does not fit; here is the mixture". (5) *The feasibility wall eats the applications* — several planned measurements return "not resolvable", which is honest but not exciting. (6) *The nGPT correspondence is scale-dependent* — it holds at p = 256 and 512 but the ratio κ̂_res/s_z drifts, and a reviewer demands the p = 1024 point that costs 100+ A100-hours. Mitigated by pre-registering the ratio-versus-p trend as the reported object (so a drift is a *result*, not a hole), by including the p = 768 point in Tier 2, and by §4.8 stating exactly what the missing run would cost. (7) *A reviewer reads the model list as stale.* Four generations of checkpoints in one table invites "why is Pythia here in 2026". Mitigated by stating the reason in the caption and in §4.0 — the audit must use the audited models, the head taxonomies exist only for the older models, Pythia's ~150-checkpoint trajectory has no modern equivalent, and the Qwen3.5 entries (4B, 9B, 27B, 35B MoE) are present precisely so the generational axis is measured rather than assumed.

**Which applications are most likely to produce a genuinely surprising result**, in order of expected value:

1. **(d) attention-head diagnostic.** Best statistical regime by two orders of magnitude (n/p ≈ 1.6 × 10⁴), least explored, immediately checkable against existing head taxonomies, and — critically for this budget — completely unaffected by the move to small models, since head dimension does not scale. Eleven models give ~1800 heads and a real ROC curve.
2. **(e) the audit.** Bounded surprise but the most ICLR-shaped contribution: a table of which published claims are null artifacts is inherently newsworthy, and the p/√n rule is a durable takeaway. Its targets are small-model studies, so it is native to this budget. Risk: nothing falls.
3. **(c) the nGPT s_z test.** Binary, pre-registerable, extremely well-conditioned under the *corrected* estimand (relative precision 10⁻⁴), interpretable in both directions, and now a three-point scaling ladder rather than a single checkpoint.
4. **(b) layerwise κ.** Likely to reproduce known depth trends with better error bars; the cross-family and now cross-generation comparison across eleven models is the part that could surprise.
5. **(a) the embedding atlas.** Most likely to hit the feasibility wall. Use as the motivating figure, not the finding. Free, so it runs regardless — and the Pythia trajectory inside it is the cheapest genuinely novel object in the section.

### 4.7 The cheap-first ladder, decision gates, and compute budget

**Tier 0 — 4.0 GPU-hours + ~62 CPU-core-hours. The go/no-go.**

| # | Item | GPU-h | CPU-core-h |
|---|---|---|---|
| T0.1 | Re-measure sampler throughput on the actual 3090; KS/AD goodness-of-fit at p ∈ {128, 1024, 4096, 8192}, all dtypes | 1.0 | 5 |
| T0.2 | Coarse bias surface: p ∈ {768, 1024, 4096}, 5 κ/p, 4 n, 200 reps; verify the κ̂_null ≈ p/√n prediction | 0 | 15 |
| T0.3 | **Full embedding atlas**, ~31 checkpoints incl. Qwen3.5-4B/9B/27B/35B-MoE, tensor-only loads, κ̂_bc + null + movMF/BIC; **plus the Pythia ~150-checkpoint trajectory** (8 of the 42) | 0 | 42 |
| T0.4 | Head + layerwise pilot: GPT-2 124M and Pythia-410m, 2M tokens, q/k/o hooks | 1.5 | — |
| T0.5 | nGPT config A pilot (p=256, 1 seed, 200M tokens) + 5-point LR sweep at 10% horizon | 1.5 | — |

**Tier 0 gates, stated as specific observations:**

- **GATE A (instrument).** *Proceed* if T0.2 reproduces κ̂_null ≈ p/√n to within 20% across all three p, and the second-moment correction drives the null κ̂ below 10% of the uncorrected value. *Stop the methods contribution* if the correction does not reduce null bias — then the paper has no corrected estimator and reduces to the CRLB negative result plus null tables, which is a workshop artifact, not ICLR.
- **GATE B (atlas / motivation).** *Proceed* if at least **6 of ~31** checkpoints return κ̂_bc whose lower CI bound exceeds the 99th null percentile (i.e. real, resolvable concentration) **and** at least **6** return "not resolvable" — that split is the motivating figure. (Threshold restated from 8-of-40 to preserve the same ~20% fraction on each side against the atlas's actual 31 entries.) *Downgrade the atlas to a one-panel figure* if essentially all 31 are unresolvable; it is still honest but it stops being a section.
- **GATE C (heads — the primary gate).** *Proceed to Tier 1* if in the T0.4 pilot the key-distribution κ̂ separates the top-decile prefix-matching (induction) heads from the bottom decile with AUC ≥ 0.70 in *both* models. **STOP and reallocate to Projects 1–3** if AUC ≤ 0.55 in both — the best statistical regime in the paper failing to see structure means κ̂ is not measuring anything a taxonomy cares about, and no amount of extra models will fix that.
- **GATE D (nGPT).** *Proceed* if the p=256 nGPT reaches a validation loss within 5% of its matched baseline at its own best LR, s_z converges to a stable value > 5, and κ̂_res is estimable with the known-mean estimator at n ≥ 10⁶ (CI width < 1% of κ̂). **STOP the nGPT line** if s_z does not converge or the tiny model's s_z is < 5, because then the p ≤ 768 ladder is measuring a degenerate regime that says nothing about nGPT at p = 1024.

**Tier 1 — 26.1 GPU-hours + ~350 CPU-core-hours. Unlocked by Gate C (or by A+B if C is borderline and B is strong).**

| Item | GPU-h | CPU-core-h |
|---|---|---|
| Full bias surface + null tables (6 × 10⁹ vectors, p ≤ 8192) — CPU, off the critical path | 0 | 200 |
| Bootstrap CI coverage study + median-unbiased calibration tables | 0 | 120 |
| CRLB / A′_p evaluation | 0 | 10 |
| Synthetic head-to-head vs cosine / Ethayarajh / IsoScore / PR / effective rank / condition number | 0 | 20 |
| (b) Layerwise κ, 11 small models × 10M tokens (incl. Qwen3.5-4B), streamed resultants | 6.5 | — |
| (d) Attention-head diagnostic, same passes + q/k/o hooks, ~1800 heads | 3.5 | — |
| (e) Audit re-implementations + document-level block bootstrap | 2.5 | — |
| (f) Vision validation: ResNet-18 CIFAR-100 × 2 seeds if not reusable, + feature extraction on CIFAR-10/100, STL-10, Tiny-ImageNet | 3.0 | — |
| (c) nGPT ladder A + B, remainder after the Tier-0 pilot: nGPT × (3, 2) seeds, matched baselines × (3, 2) seeds, per-arm LR sweeps (12.1 total − 1.5 pre-paid) | 10.6 | — |
| **Tier 1 total** | **26.1** | **350** |

**Tier 1 → Tier 2 gate.** *Unlock Tier 2* if **either** (i) the head diagnostic's induction-vs-other AUC exceeds 0.75 pooled across eleven models with per-model bootstrap CIs excluding 0.5 for the models that have the power to do so (see §4.4(d)) — in which case Tier 2 buys the p = 4096 confirmation on a 9B model that a reviewer will ask for; **or** (ii) κ̂_res/s_z lands in [0.9, 1.1] at *both* p = 256 and p = 512 with a |trend| in p under 5% per octave — in which case the p = 768 point is worth 17 GPU-h because it makes the correspondence a three-point law rather than a two-point coincidence. **Do not spend Tier 2** if the head AUC is between 0.55 and 0.75 *and* the nGPT ratio is outside [0.8, 1.25] at either size: that combination means both headline findings are mushy, and the right move is to write the methods paper plus the audit and stop.

**Tier 2 — 30.0 GPU-hours. Only if the Tier-1 gate fires.**

| Item | GPU-h |
|---|---|
| (c) nGPT config C: p = 768, 85M params, 800M tokens, nGPT + matched baseline + LR sweep | 17.0 |
| (b)+(d) **Qwen3.5-9B in NF4 4-bit**, inference only, 10M tokens, layerwise + head statistics at p = 4096 | 10.0 |
| Extended audit table, movMF/BIC at scale, final figure production | 3.0 |

The 9B NF4 pass is priced at **1.8 × 10¹⁷ FLOPs** (2ND with N = 9 × 10⁹, D = 10⁷) against **~5 TFLOPS effective** for 4-bit dequant-bound inference on a 3090 — **10.0 GPU-h**, up from the 8.0 previously budgeted for a 7–8B model, i.e. roughly the 1.25× the parameter ratio implies. Nothing about the 9B's hybrid layer stack changes this estimate: the statistics are read off hidden states and per-head q/k/o projections wherever an attention block exists, and the layers that do not expose per-head projections simply contribute layerwise κ only. Llama-3.1-8B and OLMo-2-7B remain the *fallback* choice for this slot if the current-generation checkpoint turns out to be awkward to quantise; the scientific requirement is one p = 4096 confirmation, not a specific vendor — and no training happens in this slot, so the policy does not constrain the fallback.

**Totals.**

| Tier | GPU-h | CPU-core-h |
|---|---|---|
| Tier 0 | 4.0 | 62 |
| Tier 1 | 26.1 | 350 |
| Tier 2 | 30.0 | 0 |
| **Subtotal** | **60.1** | **412** |
| 20% contingency (GPU only) | 12.0 | — |
| **TOTAL** | **≈ 72 GPU-hours** | **≈ 412 CPU-core-hours** |

Sanity check against the constraint: 72 GPU-hours is **7.2% of the 1000 GPU-hour ceiling** and **72% of the 100-hour target**. Peak VRAM: the largest training run (nGPT p=768, 85M params, ctx 512, bf16, AdamW) needs ~0.17 GB weights + 0.17 GB grads + 0.68 GB optimizer + activations, i.e. under 6 GB at batch 32 — the 3090 is not close to full. The largest bf16 inference job is Qwen3.5-4B (~8 GB of weights, forward-only, no KV cache, packed 1024-token sequences) at ~12 GB total, and the largest 4-bit job (Qwen3.5-9B in NF4) is ~6 GB of weights plus ~4 GB of activations at batch 8. **Nothing in this plan is memory-limited, and nothing in it needs the A6000** — if the A6000 happens to be idle it shortens wall-clock (larger batches on the (b)/(d) passes) without changing the GPU-hour budget, but the plan does not assume it.

**What was cut or changed, explicitly.** (1) The 500–800 H100-hour nGPT-0.5B pretrain at p = 1024 — replaced by the p ∈ {256, 512, 768} ladder at 29.1 GPU-h, which is 50–80× cheaper and adds seeds, matched baselines and LR sweeps the original could not afford; the p ≥ 1024 point is deferred to §4.8. (2) bf16 layerwise passes on Pythia-6.9B and Llama-3-8B — replaced by eleven models ≤ 4B plus one NF4 Qwen3.5-9B, which costs less and gives cross-family breadth the original lacked. (3) The bias-surface grid shrank from ~10¹⁰ to 6 × 10⁹ sampled vectors: κ/p grid 12 → 9 points, and replicates at n ≥ 10^5.5 from 50 to 50/25, costing tail-quantile precision at the largest n only. (4) The CI-coverage study runs on a reduced 5 × 6 × 6 grid rather than the full surface. (5) **Model-policy revision:** four Qwen3.5 checkpoints added to the atlas at ~2 CPU-core-hours and ~8 GB of download; the Pythia trajectory sub-study added at ~8 CPU-core-hours and ~45 GB; Qwen3.5-4B added to the forward-pass set (+1.5 GPU-h in (b), +0.5 in (d)); the Tier-2 confirmation model moved from an 8B to Qwen3.5-9B (+2.0 GPU-h). Net GPU change from the policy: **+4.0 h** before contingency. (6) **Re-costing corrections applied in the same pass, which move the total more than the policy did:** the nGPT ladder's tier roll-up previously under-counted Tier 1 by 1.6 h (12.1 h of ladder rows minus the 1.5 h Tier-0 pilot is 10.6, not 9.0), the Tier-1 CPU line under-counted by 40 core-hours (200 + 120 + 10 + 20 = 350, not 310), the 9B NF4 FLOP figure was 2× too high and its assumed throughput 2× too optimistic (both corrected; the 10.0 GPU-h stands), the atlas is ~31 checkpoints rather than ~45 (Gate B thresholds restated at the same 20% fraction), and the download estimate falls from ~375 GB to ~180 GB once shard-level selectivity is applied honestly. Subtotal moves 58.5 → 60.1 GPU-h, total 70 → 72. **No seeds were cut** — the nGPT ladder keeps 3/2/1 seeds at A/B/C, and the LR sweeps are untouched, because statistical power is the thing this program keeps getting wrong. Nothing scientific was removed: every finding in §4.4 is still reachable.

**Timeline.** Week 1: Tier 0 (the GPU work is 4 hours; the wall-clock is dominated by the ~180 GB of downloads — including ~300 Pythia revisions, which are many small requests rather than a few large ones — and by writing the streaming harness). **Gate decision end of week 1.** Weeks 2–5: Tier 1, with the 200-core-hour null-table job running on the CPU throughout and never blocking the GPU. **Tier-1 gate end of week 5.** Weeks 6–9: Tier 2 if unlocked (the p = 768 nGPT run is 17 GPU-h ≈ 1 day of wall-clock; the 9B NF4 pass is another half-day). Weeks 6–12: writing, with the paper's narrative built around whichever of {d, c, e} landed.

**The pre-committed no-go remains acceptable** because the negative result of §4.2 — "no estimator can resolve κ/p < 0.05 at n < 10⁵, with a CRLB to prove it, therefore the isotropy question is unanswerable at vocabulary scale for p ≥ 4096" — is citable on its own, and because the entire Tier-0 spend is four GPU-hours. The cost of finding out that this project does not have a paper in it is less than one afternoon of the 3090.

### 4.8 Deferred: needs language-model compute

Three items are out of the single-3090 budget. They are listed here rather than hidden in the main plan, each with its cost and the specific Tier-1/2 observation that would justify unlocking it.

**D1. nGPT at p ≥ 1024 (the dimension nGPT actually used).** A 0.5B-parameter nGPT at p = 1024, ctx 1024, ~10B tokens (Chinchilla-ish) is ~3 × 10¹⁹ FLOPs ≈ 420 GPU-h at 20 TFLOPS effective, ×1.6 for nGPT overhead ≈ **670 3090-hours** (≈ 250 A100-hours), and it does not fit in 24 GB for training without sharding or aggressive activation checkpointing at small batch — with checkpointing it is feasible in memory but the wall-clock becomes ~4 weeks of continuous 3090 time. A reduced version — p = 1024, 12 layers, ~150M params, 3B tokens — is ~**110 3090-hours** and *does* fit comfortably in 24 GB; that is the realistic unlock and it would still exceed this project's 100-hour target on its own.
**Unlock condition:** the Tier-1/2 ladder shows κ̂_res/s_z systematically drifting with p — specifically, |κ̂_res/s_z − 1| increasing monotonically across p ∈ {256, 512, 768} by more than 5% per octave. That trend, extrapolated, would put the correspondence in doubt at exactly nGPT's own p = 1024, and only a p = 1024 run can settle it. If instead the ratio is flat and inside [0.9, 1.1] at all three sizes, **do not run D1** — the correspondence is established and a fourth point adds nothing.
**Alternative that costs nothing:** if NVIDIA or a third party releases nGPT weights before submission, the test becomes a ~1 GPU-h inference job on a rented A100 (or NF4 on the 3090), and D1 disappears. Check for a release before spending anything.

**D2. Layerwise and head statistics on ≥ 13B models in bf16, and any 70B forward pass.** OLMo-2-13B (p = 5120), **Qwen3.5-27B and the Qwen3.5-35B MoE**, and Llama-3-70B (p = 8192) cannot be run forward on a 24 GB card at useful throughput even in 4-bit (70B in NF4 is ~40 GB; the 35B MoE's router makes activation memory unpredictable at batch). Cost on rented hardware: ~4 A100-80GB-hours for 10M tokens through a 13B, ~8 through a 27B, ~12 through the 35B MoE, ~30 through a 70B. **Note that all of their embedding matrices are already covered at zero cost in §4.4(a)** — only the *hidden-state* studies are deferred, and the atlas therefore already contains the current-generation entries the policy asks for.
**Unlock condition:** the Tier-2 NF4 Qwen3.5-9B run shows the layerwise κ profile changing *shape* (not just scale) relative to the ≤ 4B models — e.g. the depth at which κ peaks moving monotonically with model size. That would make the trend itself the finding and would require ≥ 2 more points to characterise, in which case **Qwen3.5-27B is the first one to buy** because it holds family, tokenizer and recipe fixed against the 4B and 9B points already in hand. If the 9B profile is a rescaled version of the 4B profile, the small-model evidence is sufficient and D2 stays deferred.

**D3. Project 3's RL diversity-collapse diagnostic at 9B+.** The κ̂-over-hidden-states diagnostic is instrument-side free, but running it on a Qwen3.5-9B RL policy requires Project 3 to have one — which is an A6000 LoRA job in Project 3's budget (base frozen, reference policy free via adapter-disable), not this one's. Deferred with Project 3. **Unlock condition:** none from this project — it rides on whatever scale Project 3 reaches.

Nothing else in this section requires compute beyond a single RTX 3090.


---

## Project 5 - Breaking the Dimension Wall in Hyperspherical VAEs: A Numerically Valid vMF Posterior at m ≥ 1024

Target venue: ICLR. Primary artifact: a reproduction-plus-extension of Davidson et al. (UAI 2018, arXiv:1804.00891) that goes where the original could not, and a falsification test of the premise behind Power Spherical (De Cao & Aziz, 2020, arXiv:2006.04437).

### 5.0 Budget headline: this project fits on one 3090 with room to spare

**Total compute: ≈72 GPU-hours of core experiments, ≈97 GPU-hours including a 35% slack allowance for failed runs and sweep overshoot. All on a single RTX 3090 (24 GB). That is under the 100 GPU-hour target and roughly 10× under the 1000-hour ceiling.**

The reason is structural, and it should be stated plainly in the paper's reproducibility section because it is a virtue: **every model in this project is an MLP VAE with under 1.5M parameters or a 2-layer GCN on a graph with under 20k nodes.** An MNIST MLP VAE at the original's architecture ([256,128] encoder, [128,256] decoder, batch 128, 300 epochs, data resident on-device) is roughly **3–5 minutes of 3090 wall-clock**, IW-500 evaluation included. A Cora/Citeseer GAE run is under a minute. The previous version of this plan quoted 0.4–0.6 GPU-h per MNIST run on an A100-class card; that was off by roughly 5×, because these models are launch-overhead-bound, not FLOP-bound, and a 3090 is not meaningfully slower than an A100 on a workload that never fills either card. Correcting the per-run cost is what turns a nominal 600-hour program into a 72-hour one — the grid barely shrank.

Consequence for planning: the whole design is **1,315 individual training runs**, with 10 seeds everywhere it matters. This is the "many small models" regime by construction, not by concession.

**Peak VRAM, worst cases (all comfortably inside 24 GB):**

| Configuration | Peak allocated |
|---|---|
| MNIST S-VAE, d=1024, batch 128, training | ~0.4 GB |
| MNIST S-VAE, d=4096, batch 128, training | ~1.1 GB |
| IW-500 evaluation at m=4096 (chunked to 64 IW samples) | ~1.6 GB |
| Pubmed S-VGAE, d_z=256, negative-sampled decoder | ~2.9 GB |
| Pubmed with the *naive* dense n² decoder (avoided; costed for the memory ablation) | ~1.6 GB decoder + tape |

Nothing here is near the limit. The **real** cost of this project is not GPU-hours at all, it is **developer-days**: §5.1's two engineering deliverables are ~4 weeks for one person. That is the schedule risk, and no amount of compute buys it down.

**Positioning: this is a co-starter with P4.** Because the GPU footprint is small and bursty (hundreds of 4-minute runs), P5 can share a single 3090 with P4's longer jobs by running in the gaps, or simply run to completion in ~4–5 days of dedicated wall-clock. Start P5's engineering in parallel with P4's experiments; the two contend for the developer, not the card.

**Calendar honesty.** Engineering is ~4 weeks and the experiments are ~1 week of wall-clock, so the full program is ~7–8 weeks, not the ~14 previously stated. The ICLR 2027 abstract deadline (~mid-September 2026) is six weeks out. Tier 0 + Tier 1 (§5.1–§5.4, §5.6) fit that window if engineering starts immediately; Tier 2 (§5.5 and the extra-dataset blocks) is the camera-ready appendix or the follow-up.

### 5.1 The two blockers, and why this repo removes them

The hyperspherical VAE needs exactly two primitives that off-the-shelf code does not provide at scale. Both are engineering, not research, and **both are measured in developer-days, not GPU-hours.**

**(a) A numerically stable, differentiable vMF→uniform KL at large m.**
For a vMF(µ, κ) posterior on S^{m-1} ⊂ R^m against the uniform prior,

```
KL = κ·A_m(κ) + log C_m(κ) + log( 2 π^{m/2} / Γ(m/2) )
log C_m(κ) = (m/2 - 1) log κ - (m/2) log(2π) - log I_{m/2-1}(κ)
A_m(κ)     = I_{m/2}(κ) / I_{m/2-1}(κ)
```

The numerator Bessel has order m/2. The kappa benchmark (Aug 2026) established that `scipy.special.ive` underflows to *exactly zero* above order ≈511, so log I becomes -inf and any derived quantity becomes NaN. Order 511 is crossed at **m = 1024**. The reference S-VAE implementation routes the Bessel ratio through `scipy.special.ive` inside a custom autograd `Function`; it therefore cannot produce a finite loss at m ≥ 1024 — not "degrades", *cannot run*. This is a reportable finding in itself and belongs in the paper's introduction.

The repo already has the replacement: compiled CUSF (~1e-13 relative error at every order tested) and a pure-PyTorch `torch_log_iv` port (uniform asymptotic / Debye expansion plus power series). What is missing is the autograd layer. Deliverable:

- A `vmf_kl(kappa, m)` op with a **hand-written backward**, not autodiff through the Debye series. The derivative is closed-form and needs no Bessel differentiation at all: the `log C` and entropy terms cancel, leaving
  `dKL/dκ = κ · A'_m(κ)` with `A'_m(κ) = 1 - A_m(κ)² - (m-1)·A_m(κ)/κ`.
  Only `A_m` must be accurate, and `A_m` is exactly the quantity the repo's κ-estimator already computes in log space. Backward cost = one extra ratio evaluation.
- A `vmf_log_prob(z, mu, kappa)` for the 500-sample importance-weighted evaluator, which needs `log C_m(κ)` itself (not just its derivative).
- fp64 for the Bessel path regardless of the network dtype; the repo's samplers already support mixed dtype. On a 3090 fp64 is 1/32 rate, but the Bessel path touches B scalars per step (128 numbers), not B×m tensors, so its cost is invisible — confirm this in the §5.4 profile rather than assuming it.
- Tests: finite differences against `mpmath` at m ∈ {2, 8, 40, 128, 512, 1024, 4096} × κ ∈ {1e-2, 1, 10, 1e2, 1e3, 1e5}, target 1e-10 relative on value and gradient; an explicit regression test asserting SciPy NaNs where CUSF does not.
- **Estimate: 6 working days. GPU cost: ~0 (CPU/fp64 validation).**

**(b) Batched per-row (µ_i, κ_i) sampling.**
`vMFSampler.__init__` stores `self.kappa = float(kappa)` and a single µ, with the Householder reflection cached per-µ. A VAE needs one sample per datapoint from a *different* posterior per row: µ ∈ R^{B×m}, κ ∈ R^B, output (B, m) or (B, K, m) for K IW samples. Deliverable:

- Vectorize the Ulrich rejection loop over rows with a masked `while` on still-unaccepted indices, index compaction each iteration, a hard iteration cap, and a deterministic fallback. Acceptance is 1.0–1.46 tries/sample per Davidson's own table, so 3–4 iterations covers essentially all rows; the cap is a safety net, not a hot path.
- Use the numerically stable form of the rejection constant: `b = (m-1) / (2κ + sqrt(4κ² + (m-1)²))` rather than the algebraically equivalent `(-2κ + sqrt(4κ² + (m-1)²))/(m-1)`, which cancels catastrophically at large κ/m.
- Per-row Householder as an in-place rank-one update (`x ← x - 2v(vᵀx)/‖v‖²`), never materializing B separate m×m rotations. This is the same primitive that gives 126.97× (NumPy) / 99.01× (Torch) over SciPy at d=4096, and it is what makes §5.6 possible. **Note the known bug:** the current `S_outer` path is not truly in-place; fix and re-verify before it is used in a training loop, since a B×m×m temporary at B=128, m=1024 is 512 MB per copy in fp32 and *would* actually threaten 24 GB.
- Reparameterized gradients: the accept-reject reparameterization of Naesseth et al. (2017) with the `g_cor` score-function correction term, as in the original S-VAE. Gradient correctness test: compare `∂/∂κ E[f(z)]` against a high-sample finite-difference estimate on a smooth `f`.
- **Estimate: 8 working days**, plus 2 days for the numerical/gradient test suite. **GPU cost: ~1 GPU-h of benchmarking, folded into T1.5.**

**(c) Harness.** Training loop, dynamic binarization on-GPU, IW-500 evaluator (chunked over IW samples so m=4096 stays under 2 GB), Power Spherical posterior, logging of rate/distortion/κ separately, a run-launcher that can keep a 3090 saturated with several concurrent 4-minute jobs. **Estimate: 5 days.** Total engineering: **≈4 weeks for one person, ≈0 GPU-hours.**

### 5.2 Faithful reproduction

Nothing is claimed until the original numbers come back. Exact target configuration from Davidson et al.:

| Item | Setting |
|---|---|
| Data | MNIST, **dynamically binarized** (resample Bernoulli each epoch) |
| Encoder / decoder | MLP [256, 128] / [128, 256], ReLU |
| Optimizer | Adam, lr 1e-3, batch 64 (batch 128 for all non-reproduction blocks) |
| KL schedule | linear warm-up over the first 100 epochs |
| Latent dims | d ∈ {2, 5, 10, 20, 40} |
| Evaluation | log-likelihood via 500-sample importance weighting, fp64 log-mean-exp |
| Seeds | 10 runs per cell, report mean ± s.d. |

Targets to hit: **d=2, LL −132.50 ± 0.73**; **d=40, S-VAE LL −90.87 ± 0.34** vs **N-VAE −88.93 ± 0.30**. Reproduction criterion: every cell within 2 s.d. of the published mean, and the *sign* of the S-VAE−N-VAE gap correct at every d (S-VAE wins low-d, Gaussian wins at d=40).

One convention trap to pin down and report explicitly: S^{m-1} ⊂ R^m has m−1 intrinsic dimensions, so a degrees-of-freedom-matched comparison pairs S-VAE at m = d+1 against N-VAE at d. Record which convention the published numbers use, and report both alignments in the appendix. Reviewers of hyperspherical papers ask this every time.

Also log, per d: rate (KL, nats), distortion (reconstruction BCE, nats), the full histogram of learned κ, and wall-clock s/epoch. The original reports LL only; the diagnostics in §5.3 are impossible without the decomposition.

**Cost: 5 d × 2 families × 10 seeds = 100 runs ≈ 8 GPU-h.**

### 5.2b The learning-rate confound, and why there is no excuse for it here

Every "normalized vs unnormalized" comparison is silently also a learning-rate comparison. The mechanism is the same one that afflicts weight normalization — when a quantity is projected back onto a fixed-norm manifold, the effective step size scales like η/‖·‖², so an arm with normalized latents sees a different effective η than an arm without, at identical nominal η. Here the S-VAE and PS-VAE decoders receive unit-norm inputs while the N-VAE decoder receives inputs of typical norm √d; at d=1024 that is a factor-32 difference in input scale and therefore in first-layer gradient magnitude. A single shared lr=1e-3 across arms does not compare posterior families, it compares posterior families *at one arbitrary point of a per-arm-optimal learning rate curve*.

**Per-arm LR sweeps are mandatory, and at this budget they are nearly free.** Each MNIST run is ~4 minutes; a 3-point log-spaced sweep per arm at two dimensions costs 5 GPU-h out of 72. There is no defensible reason to skip it, and the paper should say exactly that — a reviewer who asks "is this a learning-rate effect?" gets a figure, not a paragraph.

Protocol: lr ∈ {3e-4, 1e-3, 3e-3} × {N-VAE, S-VAE, PS-VAE} × d ∈ {40, 1024} × 3 seeds = 54 runs. Select the per-arm best lr by validation ELBO and **carry that per-arm lr into every downstream block**, reporting the selected values in a table. If the S-VAE−N-VAE gap at d=40 changes sign or loses significance under per-arm tuning, that is itself a headline finding about the original comparison, and it must be reported as such rather than buried.

**Cost: 54 runs ≈ 5 GPU-h.**

### 5.3 The central experiment: break the dimension wall

Extend the same model to d ∈ {64, 128, 256, 512, 1024} and — because the compiled Bessel path makes it possible and the budget makes it painless — **two further octaves to d ∈ {2048, 4096}**, holding everything else fixed. The original stops at 40 and attributes degradation above m≈20 to vanishing sphere surface area. Three hypotheses, each separately testable:

**H1 — sampling cost / sampler pathology.** *Prediction: refuted, and refutable using the original's own data.* Davidson report acceptance 1.0–1.46 that **improves** with dimension. Measurement: acceptance rate and rejection-loop iteration count vs d on the batched sampler, plus s/epoch decomposed into sampler / encoder / decoder / KL. If sampler time is <5% of step time at d=1024 (expected, given the measured speedups), H1 is dead and the paper says so with a number.

**H2 — the rate floor: KL growth swamps reconstruction.** Under a uniform prior, KL depends only on (κ, m) and grows roughly linearly in m at fixed κ. The original's own KL goes 7.3 → 33.5 across d=2→40. So to keep the rate bounded, the encoder must drive κ down, which destroys the informativeness of the posterior. *This is the hypothesis I expect to survive.* Tests (Tier 2, gated): (i) rate–distortion frontier per d via a β sweep, β ∈ {0.1, 0.25, 0.5, 1, 2}; (ii) free-bits floor; (iii) a fixed-κ ablation (learn µ only, κ ∈ {10, 50, 200, m/4}) which removes the encoder's ability to collapse the rate; (iv) active-unit count and mean/percentile learned κ vs d; (v) a fixed-width-decoder control so that changing d does not also change decoder capacity.

**H3 — a numerical ceiling on learned κ.** As m grows, holding rate or reconstruction fixed requires larger κ, and the Bessel ratio evaluation degrades. Test: rerun the d ∈ {64…4096} grid with three Bessel backends (CUSF, torch port, SciPy where it survives) and compare learned κ trajectories, loss values, and NaN/stall rates. Two extra dimensions past 1024 matter here: the SciPy arm is *guaranteed* dead at 1024 and beyond, so 2048/4096 are the cells where CUSF-vs-torch-port agreement is the only available cross-check.

> **Honesty constraint, to be stated verbatim in the paper.** H3 **cannot** explain the degradation the original reports at m ≈ 20–40. SciPy's `ive` underflow begins near order 511, i.e. m ≈ 1024, two orders of magnitude of dimension above where the reported degradation appears. The numerical blocker is why the original *stopped* at 40 and why nobody has run m ≥ 1024 correctly since; it is **not** the cause of the m=20–40 result. Any framing that conflates the two is overselling, and a competent reviewer will catch it.

**What the likely negative result looks like.** S-VAE at d=1024 runs cleanly, produces finite gradients, costs almost nothing extra in sampler time — and still loses to the Gaussian on LL. That is publishable, and arguably the better paper: it converts a folk belief ("vMF doesn't scale because sampling breaks") into a measured statement ("the uniform prior on S^{m-1} imposes a rate floor that grows with m; the sampler was never the problem"), and it motivates §5.5 as the fix. The unpublishable outcome is inconclusive numerics, which is what the fp64 Bessel path and the mpmath-validated gradient tests exist to prevent.

**Cost: main grid 5 d × 3 families (N/S/PS) × 10 seeds = 150 runs ≈ 15 GPU-h; extreme grid d ∈ {2048, 4096} × 3 families × 5 seeds = 30 runs ≈ 6 GPU-h.**

### 5.4 vMF vs Power Spherical at matched compute

Power Spherical exists explicitly because vMF rejection sampling was "slow and numerically unstable in high dimensions". This repo may falsify half of that premise. The head-to-head must be honest: same encoder/decoder, same d grid, same optimizer, same seeds, **per-arm-tuned lr from §5.2b**, swap only the posterior family (and its matching KL against the uniform prior, which for PS is closed-form in Beta functions and has no Bessel at all).

Report three axes, never one: **(1)** IW-500 log-likelihood; **(2)** wall-clock — s/epoch, µs per 1e6 posterior samples, peak memory; **(3)** gradient variance.

**Run the gradient-variance study first, on a synthetic objective, before any VAE training.** It is Tier 0, it is inference-only, and it costs ~2 GPU-h. Rationale: the S-VAE reparameterization carries a `g_cor` correction that is score-function-style, and score-function terms have variance that grows with dimension. PS is a pure pathwise reparameterization. If vMF loses to PS, it matters enormously *why*: a speed loss is fixed by better kernels (this repo), a variance loss is not, and the paper's thesis changes accordingly.

Protocol: objective `L(µ, κ) = E_{z~q}[f(z)]` for `f(z) = ⟨z, t⟩` and `f(z) = -‖Wz - y‖²` with a fixed random W, over m ∈ {8, 40, 128, 512, 1024, 4096} × κ ∈ {1, 10, 100, m/8, m/2}. For each cell, draw S=1000 independent single-sample gradient estimates, report per-coordinate variance, the trace of the gradient covariance normalized by ‖E[∇]‖², and for vMF the separate contributions of the pathwise term and the `g_cor` term. Ground truth for the mean gradient from 1e7 samples (at m=4096 that is 1.6e11 floats streamed in chunks, ~20 min of the 2 h budget). This single plot is likely the most cited figure in the paper, and it is the cheapest thing in the project.

Possible outcomes, all reportable: (a) vMF matches PS on LL and is now competitive or faster in wall-clock → the PS premise is obsolete for the speed half; (b) vMF is fast but higher-variance → PS's premise was right for the wrong stated reason, and the correct fix is a lower-variance vMF estimator, not a different family; (c) PS wins outright → say so, and the contribution reduces to §5.1 + §5.3, which still stands.

**Cost: 2 GPU-h (Tier 0 synthetic) + 1 GPU-h (Tier 1 wall-clock/memory profiling); the LL comparison rides on §5.3's grid at no extra cost, since PS is one of the three families.**

### 5.5 Product-of-spheres and the scalar-κ bottleneck *(Tier 2, gated)*

A single κ per datapoint is one scalar of posterior flexibility no matter how large m is. Davidson et al. 2019 (arXiv:1910.02912) decompose S^M into a Cartesian product of lower-dimensional spheres with **one κ per factor**, turning the rate into a sum of k independent per-factor KLs — each on a much lower Bessel order, hence numerically easier as well as more expressive.

**Reproduce:** static (fixed-binarization) MNIST, S^40 LL **−96.32** → 4-way product **−92.65**. 2 configs × 5 seeds = 10 runs.
**Then scale:** total latent dimension D ∈ {128, 256, 512, 1024}, factored into k ∈ {1, 2, 4, 8, 16, 32} spheres of equal dimension, 3 seeds = 54 runs (skip cells where D/k < 4). Two questions: does the product win grow with D (predicted yes, if H2 is the real mechanism), and is there an optimal factor dimension (predicted ~8–64 per factor, where each factor still has usable surface area).

This is where the batched sampler pays off structurally: k factors of dimension D/k fold into the batch dimension as a single (B·k, D/k) sampling call, so a 64-way product costs one kernel launch, not 64.

**Ablations that would kill the claim:** (i) product-of-spheres vs. a diagonal Gaussian at matched D and matched parameter count — if the Gaussian catches up, the win is "more independent rate knobs", not "spheres"; (ii) monolithic sphere with a *mixture-of-vMF* prior at matched rate — if that closes the gap, the contribution is the prior, not the product geometry; (iii) matched-rate rather than matched-β comparison, since more κ's trivially changes the achievable rate.

**Cost: 64 runs ≈ 7 GPU-h.** Seeds are cut to 3 here (vs 10 in Tier 1) — this is the block most exposed to seed noise and the first place to spend slack if the effect looks real but marginal.

### 5.6 S-VGAE link prediction, including the experiment they could not run

Davidson et al. evaluate a hyperspherical variational graph auto-encoder on Cora (2,708 nodes), Citeseer (3,327), and Pubmed (19,717), GCN encoder, 85/5/10 edge split, AUC and AP over 10 runs. Two facts to exploit: they **omitted d_z = 64 for S-VGAE because it ran out of memory**, and **Pubmed is the one dataset where their method loses**.

The OOM has a specific cause worth diagnosing rather than asserting: a per-node rotation materialized as an n×m×m tensor is 19,717 × 64 × 64 × 4 B ≈ **323 MB per copy in fp32**, multiplied by the autograd tape and by every rejection-loop iteration. The single-reflection in-place update is O(n·m) — 19,717 × 64 × 4 B ≈ **5 MB** — a ~60× reduction in the sampler's footprint, and it is the fix that unlocks the missing cell.

**This is the cheapest win in the project.** These are three tiny graphs; a full-batch 200-epoch GCN autoencoder run is 10–60 s on a 3090. The d_z=64 cell the original authors could not afford now costs about forty seconds, and even d_z=256 on Pubmed stays under 3 GB. An entire 720-run grid is 11 GPU-h.

**Honesty note that must be in the paper:** on Pubmed the *decoder* is also a memory hog (a dense n² inner-product matrix is 3.9e8 entries ≈ 1.6 GB fp32). Use negative-sampled decoding and **measure the two contributions separately**, or we will be claiming credit for a saving we did not make. Report peak allocator memory attributable to the sampler alone via `torch.cuda.max_memory_allocated` deltas around the sampling call. Note also that 1.6 GB is *not* an OOM on a 24 GB card — so the honest framing is "the original's OOM was on 2018-era hardware and an O(n·m²) sampler; we remove the algorithmic term and report both, rather than crediting the fix to a bigger GPU."

Grid: d_z ∈ {16, 32, 64, 128, 256, 512} × {GAE, VGAE (Kipf & Welling 2016), S-VGAE, PS-VGAE} × 3 datasets × 10 seeds = 720 runs. Deliverables: the missing d_z=64 S-VGAE row, the first d_z ≥ 128 hyperspherical rows, and a direct test of whether Pubmed's loss is a dimension/capacity effect that disappears once the constraint is lifted. Negative result if Pubmed still loses at every d_z: also fine, and more informative than the original's silence — it localizes the failure to the graph, not the budget.

**Cost: 720 runs ≈ 11 GPU-h.**

### 5.7 Baselines, metrics, tiered compute budget, risks

**Baselines to beat (all re-run in our harness, never quoted from papers):** N-VAE (diagonal Gaussian); S-VAE (our reimplementation, validated in §5.2); Power Spherical VAE (arXiv:2006.04437); product-of-spheres VAE (arXiv:1910.02912); mixture-of-vMF-prior VAE; fixed-κ hyperspherical VAE (ablation isolating "learned concentration" from "spherical support"); a Gaussian VAE with L2-normalized latents and no density correction (the naive practitioner baseline). Graph side: GAE, VGAE, S-VGAE, PS-VGAE. Every arm gets its own tuned learning rate per §5.2b.

**Metrics, defined:** IW-500 log-likelihood (log-mean-exp of 500 importance weights, fp64 accumulation); ELBO; **rate and distortion reported separately** in nats; active units (KL per latent factor > 0.01 nats); learned-κ distribution per d; s/epoch and µs per 1e6 samples; peak sampler memory; gradient-variance trace ratio; AUC/AP for link prediction.

**Cross-project tie-in, with the statistical caveats intact.** Report the *aggregate* posterior's fitted κ̂ against a **matched uniform null**: n uniform unit vectors in R^m, same estimator, same n, same code path. This is mandatory, not optional — the kappa benchmark showed the finite-sample bias is systematically **upward**, so "our latents are concentrated" is inadmissible without the null. Three further constraints carry over verbatim:

- **κ/p, not p, governs estimator accuracy.** Above κ/p ≈ 0.2, n = 10–25p samples give ~1% relative accuracy; below κ/p ≈ 0.05, even n = 1e5 does not. At m=1024 with a learned κ of, say, 50, κ/p ≈ 0.05 — squarely in the regime where the estimate is meaningless. **Report κ/p alongside every κ̂ and grey out cells below 0.05.** The honest statement in that regime is "indistinguishable from the matched null at this n", not a number.
- **Never report a per-datapoint estimated κ.** n=1 gives R̄=1 and a division by zero. Per-input statistics must be a **vMF likelihood** under the fitted aggregate, never a per-input κ̂. (The *encoder-predicted* κ_i is a different object and is fine to report — label it unambiguously.)
- **p ≥ 1024 requires the compiled CUSF or pure-torch log-Bessel path.** `scipy.special.ive` is exactly zero above order ~511. Any analysis script that silently imports SciPy for the aggregate-κ̂ fit at d ≥ 1024 produces NaNs or, worse, silent garbage; the harness must assert the backend.

**Tiered compute budget with decision gates. One RTX 3090; per-run costs measured, not extrapolated from A100 figures.**

| Tier | Block | Runs | GPU-h |
|---|---|---:|---:|
| **T0** | §5.4 synthetic gradient variance (inference only) | – | 2 |
| **T0** | §5.2 reproduction spot-check, d ∈ {2, 40} × {N, S} × 5 seeds | 20 | 2 |
| **T0** | §5.3 finite-loss smoke test at m ∈ {1024, 4096}, CUSF vs torch port × 3 seeds | 6 | 1 |
| | **Tier 0 subtotal** | **26** | **5** |
| **T1** | §5.2 full reproduction, 5 d × 2 families × 10 seeds | 100 | 8 |
| **T1** | §5.2b per-arm LR sweep, 3 lr × 3 families × 2 d × 3 seeds | 54 | 5 |
| **T1** | §5.3 dimension wall, 5 d × 3 families × 10 seeds | 150 | 15 |
| **T1** | §5.3 extreme d ∈ {2048, 4096} × 3 families × 5 seeds | 30 | 6 |
| **T1** | §5.4 wall-clock / peak-memory / acceptance-rate profiling | – | 1 |
| **T1** | §5.6 S-VGAE grid, 6 d_z × 4 models × 3 datasets × 10 seeds | 720 | 11 |
| | **Tier 1 subtotal** | **1054** | **46** |
| **T2** | §5.3 β sweep + free-bits + fixed-κ ablations, 3 d × 3 seeds | 81 | 8 |
| **T2** | §5.5 product-of-spheres reproduction + scaling grid | 64 | 7 |
| **T2** | Omniglot replication, 4 d × 3 families × 5 seeds | 60 | 3 |
| **T2** | Mixture-of-vMF prior + naive normalized-Gaussian baseline | 30 | 3 |
| | **Tier 2 subtotal** | **235** | **21** |
| | **Core total** | **1315** | **72** |
| | **+35% slack (failed runs, reruns, sweep overshoot)** | | **≈97** |

**Gate G0 → Tier 1.** *Spend Tier 1 if:* (i) `vmf_kl` and `vmf_log_prob` produce **finite** loss and gradients at m ∈ {1024, 4096} across 3 seeds, with CUSF and the torch port agreeing to 1e-10 on the loss and 1e-8 on `dKL/dκ`, and the SciPy path demonstrably NaN-ing at the same settings; **and** (ii) the d ∈ {2, 40} spot-check lands within 2 s.d. of Davidson's published means with the correct sign on the S-VAE−N-VAE gap. *STOP if:* the reproduction spot-check misses by more than 2 s.d. at either d — this is the single hard gate, and the correct response is to debug the harness or contact the original authors, **not** to proceed to Tier 1 and blame the discrepancy on dimension. *Also STOP (and rewrite the paper as a numerics note) if:* the torch port and CUSF disagree beyond 1e-8 at m ≥ 1024, because then there is no trustworthy reference and every downstream κ number is unverifiable.

**Gate G1 → Tier 2.** *Spend Tier 2 if* the dimension-wall grid shows **a rate floor with the predicted signature**: mean learned κ falling monotonically with d while KL still rises, active units collapsing, and the S-VAE−N-VAE LL gap widening with d beyond seed noise (≥ 2 s.d. separation at d ≥ 256, across 10 seeds, at per-arm-tuned lr). That is the observation that makes "more independent rate knobs" (§5.5) a principled fix worth testing. *STOP and write up Tier 0+1 alone if:* the S-VAE−N-VAE gap is flat in d (no widening beyond noise from 64 to 4096) — then there is no rate floor to fix, the product-of-spheres story has no motivation, and the paper is the cleaner "the sampler was never the problem, and neither is dimension" result. *Also STOP if* per-arm LR tuning erases the d=40 gap: in that case the headline becomes the §5.2b confound finding, and the whole product-of-spheres line is chasing an artifact.

**What I cut to get here.** Relative to the previous 395 A100-h (≈990 3090-h) plan: (i) the fixed-κ family was removed from the *main* dimension-wall grid and demoted to a gated Tier-2 ablation, taking that grid from 4 families × 10 seeds to 3 × 10; (ii) seeds are 5 (not 10) at d ∈ {2048, 4096} and 3 (not 10) throughout Tier 2 — the β sweep in particular went from 150 runs at 10 seeds to 81 at 3; (iii) the β/fixed-κ frontier is measured at 3 dimensions {64, 256, 1024} rather than all 5. The most significant single cut is **10-seed coverage on the Tier-2 ablation blocks**, which is a real loss of statistical power on exactly the ablations that would falsify §5.5's claim; the mitigation is that the 35% slack allowance (25 GPU-h) is earmarked first for topping those blocks back up to 10 seeds if G1 fires cleanly. Nothing scientific was removed — the grid *grew* (two extra dimension octaves, an extra d_z, Omniglot, the mandatory LR sweep) while the cost fell, because the per-run figure was wrong before.

**Risks.**

| Risk | Likely negative result | Still publishable? |
|---|---|---|
| S-VAE loses to N-VAE at every d ≥ 64 | Rate floor confirmed; sampling exonerated | **Yes** — reframe as a diagnosis of *why* hyperspherical latents stop scaling, with the product-of-spheres fix |
| PS matches vMF on LL and is faster | The 2020 premise survives on speed | **Yes**, if the gradient-variance decomposition explains it; the §5.1 numerics contribution is independent |
| §5.2 reproduction misses the targets | Blocks everything downstream | **No** — this is gate G0 and the single hard gate; budget 2 extra weeks and contact the original authors early |
| Torch Bessel port disagrees with CUSF at extreme κ/m | Restricts GPU runs to the compiled path | Mitigated: the benchmark already validated CUSF to ~1e-13; keep CUSF as the reference and gate the port on a 1e-10 test |
| Per-arm LR tuning erases the published d=40 gap | The original comparison was partly a step-size comparison | **Yes**, and it is a stronger result than the extension — but it rewrites the paper's framing, so learn it in Tier 1, not at submission |
| Pubmed still loses at d_z=64 once memory is fixed | The OOM was never the reason | **Yes**, and it is a cleaner result than the original's omission |
| Aggregate κ̂ at d ≥ 512 sits below κ/p ≈ 0.05 | The concentration analysis is uninformative at exactly the dimensions we care about | Partially — report "indistinguishable from matched null", and lean on rate/active-units as the primary diagnostic instead of κ̂ |
| Contribution reads as "engineering" to reviewers | Desk-reject risk at ICLR | Mitigate by leading with the *scientific* claim (H1/H2/H3 separation, PS premise test, LR confound), not the kernel speedups |
| §5.1's 4 developer-weeks slip | Everything slips; GPU budget is irrelevant | **The dominant schedule risk.** Compute is 72 GPU-h; the critical path is one engineer. De-risk by shipping (a) and (b) as independently testable, independently useful library PRs |

### 5.8 Deferred: needs language-model compute

**Not in this plan. Flagged, costed, and gated.**

The natural extension of a working high-dimensional vMF posterior is a **text VAE with a hyperspherical latent** — an Optimus-style architecture (Li et al. 2020) with a small encoder and a GPT-2-medium decoder, where the well-known posterior-collapse problem is precisely a rate-floor problem and where a bounded-KL spherical latent is a principled fix. This is the setting where the §5.3 result, if H2 survives, would have the most external validity: it would turn "the uniform prior imposes a rate floor" from an MNIST observation into a claim about a domain people care about.

It is deferred for two reasons. First, GPT-2-medium (355M) fine-tuning with a 512-token context, plus an encoder and the IW evaluator, is feasible on 24 GB but not comfortable, and GPT-2-large/1.5B is out entirely under the ≥1B rule. Second, and more decisively, a text-VAE result is only interesting *conditional on* §5.3 and §5.5, and running it early would consume the majority of the budget answering a question we have not yet earned.

- **Cost if unlocked:** GPT-2-small decoder, Penn Treebank + Yelp (the standard Optimus benchmarks), 3 latent families × 3 latent dims × 3 seeds = 27 runs at ~6–9 GPU-h each ≈ **200–250 GPU-h**, plus ~1 developer-week to port the harness. With GPT-2-medium instead, ≈ 500–600 GPU-h and gradient-checkpointing required to fit 24 GB. Either way this is a **separate project-sized budget**, not an appendix.
- **Result that justifies unlocking it:** gate G1 fires *and* §5.5 shows the product-of-spheres factorization recovering the lost LL at D ≥ 512 (i.e. the rate floor is both real and fixable). If the fix does not work on MNIST, it will not work on text, and the deferred run should never happen.
- **Result that permanently kills it:** the dimension-wall gap turns out to be flat in d, or PS wins outright on both LL and variance — in either case the interesting object is not the vMF posterior and the language compute would be spent confirming someone else's distribution.


---

## Project 6 — Directional analysis of classifier latents, and adversarial robustness on the hypersphere

> **Compute envelope for this project: ~99 GPU-hours on a single RTX 3090 (24 GB).**
> Hard ceiling 1000; target <100. The previous version of this section cost ~2,400 A100-hours,
> which at the 1 A100-h ≈ 2–3 RTX-3090-h conversion is **~6,000 3090-hours — roughly 60x over
> target**. Everything below is the re-scoped replacement. What was cut is listed in §6.8.
> The project is a three-tier ladder with explicit decision gates; **Tier 0 is 9.9 GPU-hours** and
> decides whether the rest is worth running at all — including, now, whether §6.6's robustness
> claim is a gradient-masking artifact (screened *before* any adversarial training is bought) and
> whether §6.5's latent certificate is arithmetically vacuous (screened in week 1, for 0.2 h,
> *before* any derivation work or encoder training is bought).

### 6.1 Framing: kappa as a better-specified NC1

Neural collapse (Papyan, Han & Donoho, arXiv:2008.08186) says that in the terminal phase of training (TPT), within-class variability collapses (NC1), class means form a simplex ETF (NC2), the classifier becomes self-dual (NC3), and prediction reduces to nearest class mean (NC4). NC1 is conventionally reported as `tr(Sigma_W Sigma_B^+) / C` — not comparable across widths, no likelihood, no null, and it needs a pseudo-inverse of a `p x p` matrix that is singular whenever `n_class < p`.

On L2-normalized features the principled directional form of "within-class variability" is the vMF concentration: fit `vMF(mu_c, kappa_c)` per class, and `kappa_c -> infinity` **is** NC1. This buys three things the trace ratio lacks: (i) a proper likelihood, so classes and layers compare by held-out log-likelihood; (ii) `kappa/p` as a dimension-comparable scale; (iii) a computable **null** — `kappa_hat` on `n` uniform directions in `R^p` is not zero, and the finite-sample bias is systematically upward, pointing the same way as the claim we want to make. That last point is enforced throughout: **no null, no number.**

**Prior art, adversarially.** Not virgin ground. CIDER (arXiv:2203.04450, ICLR'23) uses a vMF-motivated compactness/dispersion loss. KappaFace (arXiv:2201.07394) already uses **per-class fitted kappa as a class-difficulty signal** to modulate angular margins — the closest prior work to 6.2's difficulty hypothesis, and we must cite it rather than let a reviewer find it. NECO (arXiv:2310.06823) and arXiv:2502.10691 tie neural collapse to OOD detection. Xu et al. (arXiv:2306.03440, ICML'23) propose VCI, an invariant NC1 alternative — a direct competitor and the correct head-to-head baseline.

Unclaimed: kappa as a *diagnostic with a calibrated null* rather than a *loss*; the per-layer, per-epoch, **per-dimension** atlas; and an explicit admissibility boundary. **Novelty budget: the metric framing alone is thin — the contribution must be the atlas, the `p`-sweep, and that boundary, not "vMF concentration measures NC1."** On NC2 a directional treatment adds little, since the simplex ETF is already a statement about class-mean angles (`cos = -1/(C-1)`); we report the mean-direction Gram matrix against the ETF prediction as a control and claim nothing new.

**The re-scope actually helps the science.** At small scale we can afford what the flagship version could not: 3 seeds everywhere, a per-arm learning-rate sweep on every normalized-vs-standard comparison *with a 3-seed confirmation at each arm's tuned setting*, and a genuine sweep over feature dimension `p` on a fixed backbone. Breadth and controls, not scale, are what make the `kappa/p` claim checkable.

### 6.2 The kappa atlas — Tier 0 entry point

This is the cheapest and safest part of the project and it goes first.

**Models / dims (all fit comfortably in 24 GB):** MLP 3x2048 (~9M, `p=2048` — exercises the Bessel-underflow path), ResNet-18 (11.7M, `p=512`), ViT-S/16 CIFAR variant (22.1M, `p=384`), plus a ResNet-18 with a linear projection head of width `p in {64,128,256,512,1024}` used for the dimension sweep, plus normalized counterparts (nMLP / nCNN / nViT, nGPT-style, arXiv:2410.01131) shared with the program's other projects. **Datasets:** CIFAR-10, CIFAR-100 (+ its 20 official superclasses), Tiny-ImageNet (200 classes, 64x64). Per-class, per-layer, per-checkpoint `kappa_hat`, tracked across training including TPT.

**Feasibility arithmetic — done explicitly.** The estimator's accuracy is governed by `kappa/p`, and via Banerjee `kappa0 = Rbar(p - Rbar^2)/(1 - Rbar^2)` this maps to a threshold on `Rbar` alone. Solving `kappa/p = 0.2` gives **`Rbar = 0.193`**; `kappa/p = 0.05` gives **`Rbar = 0.050`**. The operating rule for the entire project:

> Report `Rbar` always (it is directly estimable at any `n`). Report `kappa_hat` only where `Rbar > 0.19`. Between 0.05 and 0.19, report `kappa_hat` with a simulation-calibrated CI. **Below `Rbar = 0.05`, kappa is not estimable at any `n` we can afford — do not report it.**

Matched-null values, `Rbar_null ~ n^(-1/2)`, `kappa_null ~ p/sqrt(n)`:

| setting | p | n/class | n/p | null `kappa_hat` | per-class fit? |
|---|---|---|---|---|---|
| ResNet-18, CIFAR-10 train | 512 | 5000 | 9.8 | 7.2 | yes (at the 10p edge) |
| ViT-S/16, CIFAR-10 train | 384 | 5000 | 13.0 | 5.4 | yes |
| RN-18 + 128-d head, CIFAR-10 train | 128 | 5000 | 39.1 | 1.8 | yes, comfortably |
| ResNet-18, CIFAR-10 test | 512 | 1000 | 2.0 | 16.2 | underpowered; bootstrap CI only |
| MLP 3x2048, CIFAR-10 train | 2048 | 5000 | 2.4 | 29.0 | **no** — `Rbar` only |
| ResNet-18, CIFAR-100 train | 512 | 500 | 0.98 | 22.9 | **no** |
| ViT-S/16, CIFAR-100 train | 384 | 500 | 1.3 | 17.2 | **no** |
| ResNet-18, Tiny-ImageNet train | 512 | 500 | 0.98 | 22.9 | **no** |
| CIFAR-100 pooled to 20 superclasses | 512 | 2500 | 4.9 | 10.2 | marginal — CI only |
| CIFAR-100 superclasses + 128-d head | 128 | 2500 | 19.5 | 2.6 | **yes** |

**Not statistically feasible, and we say so:** per-class kappa on CIFAR-100 and Tiny-ImageNet at `p >= 384`, and on any `p=2048` representation at CIFAR-scale `n`. The 10–25p rule wants 3,840–12,800 samples/class at `p=512`; CIFAR-100 has 500. Mitigations in order: (1) **pool into superclasses** — CIFAR-100's 20 official superclasses give 2,500/class, and combined with a 128-d projection head reach `n/p = 19.5`, comfortably feasible; (2) report `Rbar` with a bootstrap CI and skip kappa entirely; (3) train split, never test. **Rejected**: augmentation-resampled features (correlated draws inflate `Rbar`, biasing kappa upward — the same direction as the hypothesis) and random projection of a trained feature (changes the geometry being measured; the projection-head sweep instead *trains* at each `p`, which is the honest version).

ImageNet-1k and BREEDS are out of scope at this budget — which removes the previous version's worst feasibility problem rather than creating one. The honest finding survives intact and generalizes: **per-class NC1-as-kappa is unmeasurable in exactly the regimes the neural-collapse literature reports it most confidently in (many classes, wide features, few images per class).** That statement is now demonstrated on CIFAR-100/Tiny-ImageNet, where we can also *show* the fix (superclass pooling + narrow head) rather than only propose it.

Early layers and early epochs sit near the null (`Rbar ~ 0.014` at n=5000, p=512), below 0.05: **the early portion of every training trajectory is a region where kappa is undefined in practice.** The atlas plots `Rbar` there and switches to kappa at threshold crossing — itself a measurable onset-of-collapse marker, so this is a finding rather than a gap.

**Numerics:** `scipy.special.ive` underflows to exactly zero above order ~511, so at `p >= 1024` SciPy returns NaN and safeguarded Newton silently stalls at ~1e-3. The MLP (`p=2048`) and the 1024-d projection-head arm **must** use the compiled CUSF path or the torch log-Bessel port; we standardize on the repo path everywhere and ship a regression test that asserts finite `kappa_hat` at `p in {1024, 2048, 4096}`.

**Engineering:** `Rbar` needs only a running sum of unit vectors — `O(p)` memory, one pass, no feature dumps. At CIFAR scale we *could* dump features (50k x 512 fp16 = 51 MB), but the streaming path is kept because it is what makes ~290 checkpoints x 6 layers affordable and because it is the path the deferred language experiments would need.

**Analyses:** (a) monotonicity and saturation of `kappa_hat(t)` through TPT; (b) normalized vs standard at matched clean accuracy **and per-arm-tuned LR, confirmed at 3 seeds** (see 6.6); (c) per-class `kappa_hat` vs per-class test accuracy, ECE-15, and per-class robust accuracy (Spearman rho, permutation p-values, benchmarked against KappaFace); (d) label-free C-component vMF mixture (spherical EM), scored by ARI / Hungarian-matched accuracy; (e) **`p`-sweep**: does the `Rbar > 0.19` admissibility boundary predict, across `p in {64,...,1024}` at fixed backbone and fixed `n`, exactly where `kappa_hat` decouples from its matched null? This is the direct test of "`kappa/p` governs accuracy, not `p`", it costs 7 GPU-h, and it is the single most defensible new measurement in the section.

### 6.3 Does kappa detect adversarial inputs?

**Attacks (re-scoped):** FGSM; PGD-Linf (`eps=8/255`, `alpha=2/255`, 20 steps, 1 restart for the sweep, 100 steps / 5 restarts only for the final headline cell); PGD-L2 (`eps=0.5`); C&W-L2 (arXiv:1608.04644) reduced to **5x binary search, 200 steps** (from 9x/1000 — the extra search levels move AUROC by <0.005 in published ablations and cost 9x); AutoAttack (arXiv:2003.01128) on the **standard 1000-image test subset**, never the full 10k.

**The n=1 trap, stated plainly.** For a single input `Rbar = 1` exactly, so Banerjee gives `kappa0 = (p-1)/0` — a literal division by zero. **Per-input `kappa_hat` is not a quantity; it does not exist.** The per-input statistic must be a *likelihood or angular distance under a vMF fitted on clean training data*. Batch-level `kappa_hat` (clean vs adversarial batches, `n >= 10p`, i.e. `n >= 5120` at `p=512` — so batch-level tests run only on CIFAR-10 train-scale pools) is a separate and legitimate experiment; we run both and never conflate them.

Second honesty note: under a fitted `vMF(mu_c, kappa_c)`, `log p(z) = kappa_c mu_c^T z + log C_p(kappa_c)` is affine in cosine similarity for a *fixed* class. So the per-input score is cosine-to-class-mean with a principled cross-class calibration; the novelty is the calibration and the multi-layer aggregation, not the score. Say so before a reviewer does.

**Baselines:** MSP (arXiv:1610.02136), ODIN (arXiv:1706.02690), Mahalanobis (arXiv:1807.03888), energy (arXiv:2010.03759), and a 2025 layer-inconsistency detector (arXiv:2505.12586). **Metrics:** AUROC, AUPR, TPR@FPR=5%, FPR@TPR=95%. Plus an *adaptive* attack maximizing classification loss subject to high vMF log-likelihood (100-step PGD on `L_CE - lambda * log p_vMF`, `lambda` swept over 5 values) — a detector not evaluated adaptively is not evaluated.

**Efficiency:** the vMF score is `O(p)` per class per sample; Mahalanobis is `O(p^2)` and needs an invertible covariance — at `p=512` with 500 samples/class (CIFAR-100, Tiny-ImageNet) the class-conditional covariance is singular, which is why Lee et al. pool a shared one. The directional score has no such constraint. **This is the setting where the efficiency argument is actually demonstrable at our budget**, and it is a better demonstration than the ImageNet `p=2048` version would have been, because here we can show Mahalanobis *breaking* rather than merely being slower.

### 6.4 vMF noise injection as a defense and as an instrument

On a normalized latent, Gaussian noise is the wrong geometry: it moves off the sphere, and after renormalization its angular magnitude scales as `sigma sqrt(p-1)` — dimension-dependent and uncontrolled. vMF gives one dial with a closed-form readout, `E[mu^T z] = A_p(kappa)`.

**Matched-displacement comparison — the experiment that makes the claim non-trivial.** Match expected cosine, not noise scale. For renormalized Gaussian `E[cos] ~ 1/sqrt(1 + sigma^2 (p-1))`, so `sigma = sqrt((A^-2 - 1)/(p-1))`. At `p=512`: `A = 0.99` needs `kappa ~ (p-1)/(2(1-A)) = 25,550` and `sigma = 6.3e-3`; `A = 0.9` needs `kappa ~ 2,555`, `sigma = 2.1e-2`; `A = 0.5` needs `kappa ~ 341`, `sigma = 7.7e-2`. Every Gaussian-vs-vMF cell runs at matched `A`. Three displacement levels (`A in {0.99, 0.9, 0.5}`) x two noise families x train-time and test-time is 6 training runs plus 6 eval configs — 6.0 GPU-h total.

**Calibrated variant:** inject at `c * kappa_hat_layer`, `c in {0.5, 1, 2, 10}`, from 6.2's per-layer estimates, so the perturbation is scale-matched to each layer's own geometry. Test-time injection is evaluated **with EOT (>= 20 samples)** without exception; a stochastic defense evaluated without EOT is invalid.

**Blocking deliverable:** per-sample noise around per-sample directions needs **batched per-row `(mu_i, kappa_i)` sampling**, which the repo lacks (scalar kappa, single mu). ~1–2 days of engineering: broadcast the Wood rejection sampler over a `kappa` vector and batch the Householder reflection over a `(B, p)` mu matrix. Note the known in-place-reflection bug in the current `S_outer` path must be fixed as part of this. Nothing in 6.4 or 6.5 runs before this lands. GPU cost is negligible (0.2 h of tests); the cost is calendar time, so it starts on day 1 in parallel with Tier 0.

### 6.5 Randomized smoothing on the sphere (restricted, gated, and audited)

Cohen et al. (arXiv:1902.02918) certify `R = (sigma/2)(Phi^-1(pA) - Phi^-1(pB))` under Gaussian smoothing. The vMF density is monotone in `mu^T z`, so the Neyman–Pearson region between `vMF(mu1, kappa)` and `vMF(mu2, kappa)` is a half-space in `(mu1 - mu2)^T z` and Cohen's two-point argument should carry to a **geodesic** certificate. The 1-D marginal of `t = mu^T z` (density proportional to `(1-t^2)^((p-3)/2) e^(kappa t)`) has no closed-form CDF, so the bound needs numerical inversion — tractable on CPU, but a derivation risk.

**This subsection was rewritten after the adversarial review. Two risks were identified; both are now priced, pre-registered, and gated *before* the spend rather than after.** Risk A: a latent-space geodesic certificate composes to a *vacuous* input-space radius unless the encoder is Lipschitz-constrained. Risk B: arXiv:2108.00491 already performs latent-space smoothing with Gaussian noise on Lipschitz-constrained encoders, so our delta is the noise *geometry* alone, and must be shown to be worth something at matched displacement.

**Risk A, quantified before any GPU spend.** Composition gives input L2 radius `~2 sin(theta/2)/L` for encoder Lipschitz constant `L`. On CIFAR the reference target is the Cohen-style `0.25`–`0.5` L2 ball (for orientation, a worst-case Linf `8/255` perturbation on 3x32x32 has L2 norm `1.74`). For a standard ResNet-18 encoder the *empirical lower* bound on `L` from gradient-norm maximization is order `1e2`–`1e3`, and the layerwise-product *upper* bound is worse than `1e6`. The best latent geodesic radius reachable at `N = 1e4` is `theta ~ 0.1`–`0.3` rad, so the composed input radius is `~1e-3` L2 — **two to three orders of magnitude short of a meaningful ball.** That is not a certificate, and we will not print it as one. On a 1-Lipschitz encoder (orthogonal convs + GroupSort, per arXiv:2108.00491) `L = 1` by construction and the composed radius is `~theta ~ 0.1`–`0.3`, which *is* in range — but such encoders cost roughly 15–20 clean-accuracy points on CIFAR-10, and that trade must be reported in the same table as the radius.

**Standing publication rule, pre-registered.** No composed input-space radius below **0.25 L2** is ever printed as a certificate, in a table, an abstract, or a figure caption. Below that threshold the number is reported — if at all — as a latent-stability measurement with the word "certificate" absent, echoing arXiv:2506.13024.

**Consequently the ordering of responses is inverted from the previous draft.** (1) **Input-space vMF smoothing over normalized image directions is the primary route** — it yields a genuine, uncomposed input-space certified set directly comparable to Cohen-Gaussian at matched displacement, and it needs no Lipschitz assumption at all. (2) Latent-space vMF smoothing is run **only on the Lipschitz-constrained encoder**, where the composition is non-vacuous, and it is positioned explicitly as a geometry swap against arXiv:2108.00491's Gaussian latent smoothing, not as a new setting. (3) Latent smoothing on a *standard* encoder is reported, if at all, under the standing publication rule above — never as a robustness claim.

**Explicit budget call: certified smoothing is kept, but restricted, audited, and triple-gated.** Cohen's protocol at `n0 = 100` selection + `N = 100,000` estimation samples on 500 images is `5e7` network evaluations per cell. On a 3090 a ResNet-18 CIFAR forward runs at roughly 1e4 img/s in bf16, so one Cohen-standard cell is ~1.4 GPU-h and a 12-cell sweep is ~17 h — affordable in absolute terms but a poor use of a fifth of the whole budget for the highest-risk item in the section. We therefore run **200 test images, `n0 = 100`, `N = 10,000`**, 3 `kappa` values x {input-space, latent-on-Lipschitz-encoder} x 1 model, plus matched Cohen-Gaussian baselines: **2.0 GPU-h**, on top of **0.2 GPU-h** for the Lipschitz audit of the standard encoder (**now a Tier-0 line item, T0.7, so the threat-model question is settled in week 1**) and **1.5 GPU-h** to train the 1-Lipschitz encoder, for **3.7 GPU-h total**. The cost of the restricted protocol is stated in the paper: at `N = 1e4` the Clopper–Pearson lower bound on `pA` saturates near 0.9995, which caps the maximum certifiable radius at roughly 60% of the `N = 1e5` ceiling, and 200 images gives certified-accuracy error bars of about ±3.5 points (`sqrt(0.25/200) = 0.035`, worst-case binomial SE). **We report certified accuracy at radius and ACR with those error bars and do not compare head-to-head against leaderboard numbers computed at `N = 1e5` on 500 images.** SmoothAdv (arXiv:1906.04584) and denoised smoothing (arXiv:2003.01908) remain **cut** — both require training or fine-tuning additional models for a comparison we cannot make at matched `N` anyway.

**Gate G2a — non-vacuity precheck. Week 1, 0.2 GPU-h, runs inside Tier 0 (T0.7), before any derivation work and before the 1-Lipschitz encoder is trained.** Measure the standard encoder's Lipschitz constant (gradient-norm lower bound over 500 inputs + layerwise product upper bound) and compute the composed input radius at the best `pA` achievable at `N = 1e4`. **Decision rule, pre-registered: a latent-route certificate is reported as an input-space security claim only if the composed radius exceeds 0.25 L2. Otherwise the standard-encoder latent route is dropped on the spot**, with no appeal and no "but the upper bound is loose" rescue, and the item reduces to the input-space route plus the Lipschitz-encoder route. We expect this gate to fail for the standard encoder; failing it in week 1 is the point, and it costs 0.2 h to learn.

**Gate G2b — derivation gate. Desk work, 0 GPU-h, hard deadline end of week 2, with a week-1 tripwire.**
- *Week-1 tripwire (free):* the numerical inverter for the 1-D marginal CDF of `t = mu^T z` must agree with a `1e7`-sample Monte-Carlo estimate to `1e-6` at `p in {128, 512}` and `kappa in {10, 341, 2555}`. If the inverter is not stable by end of week 1, the whole item stops there — the multivariate bound cannot be trusted on top of an unstable scalar quadrature.
- *Week-2 gate:* the binary geodesic certificate must be derived and numerically validated (Monte-Carlo agreement with the analytic bound to within `1e-3` on synthetic vMF pairs at `p in {128, 512}`). Multi-class is *not* required to close — a binary bound plus an explicitly-stated one-vs-rest reduction is acceptable; failure to close even binary is not.
- If either fails, drop the whole item. It costs 0 GPU-h to skip, T2.6b is never trained, and the 3.5 remaining GPU-h go to a third Tier-1 seed on Tiny-ImageNet plus a second `p`-sweep seed.

**Gate G2c — delta gate, evaluated on the 2.0 h run itself, pre-registered with an explicit MDE.** vMF geodesic smoothing versus Gaussian latent smoothing (arXiv:2108.00491) **at matched expected displacement `A_p`**, reusing 6.4's matched-`A_p` machinery so the comparison is the same one the rest of the section makes. Because both smoothers are evaluated on the *same* 200 images, the test is paired, which is materially sharper than the ±3.5-point unpaired band: at an expected discordance of ~15% (`m ~ 30` discordant pairs), an exact McNemar/sign test at `alpha = 0.05` and 80% power needs `|b - c| >= 12`, i.e. a **paired MDE of ~6 points of certified accuracy at radius**. ACR is continuous and paired, with a correspondingly smaller **paired MDE of ~0.02 ACR** at the same power. **Decision rule: if vMF beats Gaussian by less than the paired MDE on both certified accuracy and ACR, the geometry buys nothing and the item ships as a single honest paragraph**, not a section. Pre-registered either way, and the paragraph is written before the run so there is no incentive to relitigate the threshold afterwards.

### 6.6 Do hyperspherical architectures resist attack better?

Measure AutoAttack robust accuracy of normalized vs standard variants at **matched clean accuracy, matched training compute, and per-arm-tuned learning rate**.

**Stated up front, because the adversarial review is right about it: this subsection is a strong a-priori candidate for a gradient-masking artifact.** Normalization bounds logits and shrinks gradients, breaking gradient attacks without conferring robustness — the canonical failure mode (Athalye et al., arXiv:1802.00420). The base rate for "normalization confers robustness" surviving Square Attack and EOT is low. The structural fix relative to the previous draft is that **the masking screen has been moved out of Tier 2 and into Tier 0 (T0.6, 0.9 GPU-h), where it runs on the plain T0.3 models — including a genuine black-box attack — before a single GPU-hour of adversarial training is committed.** Discovering masking after paying 18.2 h for T2.1–T2.3 would be a budgeting error, not a finding.

**The effective-learning-rate confound is the second threat, and at this budget there is no excuse for ignoring it.** Weight normalization makes the effective step size scale like `eta / ||w||^2`, so *every* normalized-vs-unnormalized comparison in the literature is silently also a learning-rate comparison. T0.4 therefore has two stages: a 5-point log-spaced LR sweep per arm at 60 epochs (1 seed, 2.0 GPU-h), **followed by a 3-seed confirmation run at each arm's selected LR** (60 ep, 2 arms x 3 seeds, of which 2 runs already exist in the sweep: +0.8 GPU-h). The confirmation stage is not optional: selecting a best-LR from single-seed curves and then comparing the two winners is exactly the winner's-curse error this program keeps making. **Power at the confirmation stage:** with 3 seeds per arm and a seed-level SD of 0.3 points clean / 0.5–0.8 points robust accuracy on CIFAR-10, a two-sample t-test at `alpha = 0.05` two-sided and 80% power requires Cohen's `d ~ 3.1`, i.e. an **MDE of ~0.9 points clean and ~1.6–2.5 points robust accuracy**. Any normalized-vs-standard gap smaller than that is not reportable at this seed count, and we say so rather than reporting it. Every normalized-vs-standard number in this project is reported at each arm's own tuned LR, and we additionally report the *shape* of the LR-vs-accuracy curve per arm, because a shifted optimum is the diagnostic signature of the confound. If the normalized arm's advantage disappears once each arm gets its own LR, **that is the headline result**, and it is a cheap, clean falsification of a claim that is repeated widely and tested rarely.

**Tier-0 masking pre-screen (T0.6, 0.9 GPU-h), on {std, norm} x 3 seeds, CIFAR-10, non-AT models.** Five cheap diagnostics with **numeric pass thresholds fixed in advance**, all of which are necessary conditions rather than sufficient ones:

- **(d1)** unbounded `eps` drives accuracy to **<= 0.5%** (not merely "low");
- **(d2)** robust accuracy is non-increasing across `eps in {2,4,8,12,16}/255`, with no upward step exceeding the seed CI;
- **(d3)** PGD iteration plateau: robust accuracy at 10/20/50/100 steps decreases monotonically and **PGD-100 is within 1.0 point of PGD-50** (no plateau, or continued large gains, is a fail);
- **(d4)** transfer from an undefended surrogate is **not stronger** than white-box PGD on the same model — transfer robust accuracy must be `>= ` white-box robust accuracy `- 1.0` point;
- **(d5)** **Square Attack (arXiv:1912.00049) at a reduced 500-query budget** on the 1000-image subset — black-box robust accuracy must be `>= ` white-box PGD-100 robust accuracy `- 1.0` point. This is the single most diagnostic masking signature and it costs ~0.3 GPU-h across 6 configs (500 queries x 1000 images x 6 configs at ~1e4 img/s), so running the Athalye checklist in Tier 0 *without* a black-box attack would be a false economy.

**Day-3 tripwire (free, inside the 0.9 h).** Diagnostics (d1) and (d4) are the two cheapest and the two most likely to fail. They run on the *first completed seed* of T0.3, before the remaining seeds finish. If the normalized arm fails either on that first seed, the remaining T0.3 seeds still run (they are needed for §6.2) but T2.1–T2.3 are provisionally frozen pending the full 3-seed screen — so the masking verdict starts arriving on day 3, not at the end of Tier 0.

These five feed gate G0(d) below.

**Full masking checklist (T2.5, 3.2 GPU-h across 8 AT-trained configs), retained and non-negotiable.** The Tier-0 screen is a subset run at reduced query budget on different (non-AT) models, so it does not substitute: (a) transfer from an undefended surrogate; (b) Square Attack, **5,000 queries** — black-box below white-box is a masking signature; (c) BPDA, and **EOT with >= 20 samples wherever test-time vMF injection is active**; (d) unbounded `eps` to 0%; (e) `eps`-monotonicity; (f) iteration plateau; (g) fp32 with DLR/margin loss, since bounded cosine logits underflow CE gradients in fp16. **Any claim here that skips these is worthless — we say so in the paper and report both the Tier-0 screen and the full checklist as tables, pre-registered, either way.**

**Adversarial-training baseline: Fast AT (FGSM-RS, Wong et al., arXiv:2001.03994), not Madry PGD-10.** Justification: standard PGD-10 AT is ~10-11x clean training cost; FGSM with random start plus a cyclic LR schedule is ~2x clean cost per epoch *and* converges in ~50 epochs rather than 200, so it lands at roughly 0.7 GPU-h per ResNet-18/CIFAR-10 run against ~7 h for the PGD version. That is the difference between 3 seeds x 2 arms x 2 datasets being affordable (13.2 h) and not (130+ h). Fast AT's known failure mode is **catastrophic overfitting**, so we (i) monitor PGD-10 robust accuracy on a held-out 500-image slice every epoch and early-stop on collapse, exactly as Wong et al. prescribe, and (ii) run one **PGD-7 AT control** (both arms, 1 seed, CIFAR-10, 5.0 h) to confirm the Fast-AT arms land within a few points of the full-strength recipe. If the control disagrees by more than 5 points AA, we report Fast AT results as *lower bounds only*. TRADES is **cut** — it is a third AT recipe answering a question we are not asking.

**Baselines from RobustBench, at the model scale we can actually run.** The previous version's WRN-70-16 / Swin-L table is irrelevant here — we are not competing with it and should not print it. The right reference points are the ResNet-18-scale leaderboard entries (numbers below are the ones to look up and **must be re-verified against the current leaderboard before publication**, not quoted from memory):

| leaderboard | entry class | approx clean | approx AA |
|---|---|---|---|
| CIFAR-10 Linf 8/255 | Sehwag et al. RN-18 (arXiv:2104.09425) | ~84.6 | ~55.5 |
| CIFAR-10 Linf 8/255 | Addepalli et al. RN-18 (arXiv:2210.15318) | ~85.7 | ~52.5 |
| CIFAR-10 Linf 8/255 | Wong et al. Fast AT, PreActRN-18 (arXiv:2001.03994) | ~83.3 | ~43.2 |
| CIFAR-100 Linf 8/255 | Addepalli et al. RN-18 | ~65.5 | ~27.7 |
| CIFAR-100 Linf 8/255 | Rice et al. PreActRN-18 (arXiv:2002.11569) | ~53.8 | ~19.0 |

Our Fast-AT arms should land near the Wong row, not the Sehwag row (which uses proxy-distribution extra data). We will not beat any of these. The comparison is normalized-vs-standard *at matched budget and matched recipe*, and the paper must say so plainly in the caption, not only in the text.

### 6.7 Consolidated table

| Axis | Choices |
|---|---|
| Datasets | CIFAR-10, CIFAR-100 (+ 20 superclasses), Tiny-ImageNet (200 cls, 64x64) |
| Models | MLP 3x2048 (9M, p=2048); ResNet-18 (11.7M, p=512) + projection heads p in {64,128,256,512,1024}; ViT-S/16 CIFAR (22.1M, p=384); + normalized (nGPT-style) variants of each; 1-Lipschitz encoder (orthogonal conv + GroupSort, RN-18 scale) for §6.5 only |
| Out of scope | ImageNet-1k, ResNet-50, ViT-B/16, WRN-28-10 / WRN-70-16, BREEDS — none fit the budget |
| Directional metrics | `Rbar`; `kappa_hat` (Banerjee / Sra-2-Newton / safeguarded) with matched null; `A_p(kappa)`; per-class vMF log-lik; vMF-mixture ARI |
| NC baselines | `tr(Sigma_W Sigma_B^+)/C`, VCI, ETF Gram deviation |
| Attacks | FGSM, PGD-Linf/L2 (20 steps sweep, 100/5-restart headline), C&W (5x200), AutoAttack std on the **standard 1000-image subset**, Square 500 (Tier-0 screen) and Square 5k (Tier-2 checklist), transfer, adaptive-vs-detector, EOT-20 |
| Detector / defense baselines | MSP, ODIN, Mahalanobis, energy, arXiv:2505.12586; Gaussian injection at matched `A_p`; **Fast AT (FGSM-RS)** with a PGD-7 control; Cohen Gaussian smoothing at matched restricted `N`; Gaussian latent smoothing (arXiv:2108.00491) at matched `A_p` |
| Metrics | clean/robust acc; AUROC, AUPR, TPR@5%FPR; certified acc@r + ACR (with N=1e4 caveat, paired across smoothers); composed input-space radius vs encoder `L`; per-class ECE-15; Spearman rho + permutation test |
| Seeds | 3 seeds on every CIFAR-10/CIFAR-100 arm, **including the LR-confirmation stage**; 1–2 on Tiny-ImageNet and ViT-S; every reported gap carries a seed-level CI |

### 6.8 Compute budget — single RTX 3090, tiered with gates

All figures are **RTX 3090 hours**, bf16/AMP, batch sizes chosen to fit 24 GB (ResNet-18 CIFAR at 512; ViT-S/16 CIFAR at 256 with grad-checkpointing off — peak ~9 GB).

**Tier 0 — go/no-go. 9.9 GPU-h.**

| # | Item | Est. |
|---|---|---|
| T0.1 | Matched-null calibration: 1e4 MC replicates of `n` uniform directions in `R^p`, `p in {64,...,2048}`, `n in {100,...,50000}`; tabulated null quantiles; `ive` underflow regression tests at `p >= 1024` | 0.5 |
| T0.2 | MLP 3x2048 CIFAR-10, {std, norm} x 3 seeds, 100 ep | 0.6 |
| T0.3 | ResNet-18 CIFAR-10, {std, norm} x 3 seeds, 200 ep, checkpoint every 5 ep | 4.2 |
| T0.4 | **Per-arm LR sweep** (5 log-spaced LRs x 2 arms x 1 seed, 60 ep — 2.0) **+ 3-seed confirmation at each arm's selected LR** (60 ep, +0.8) | 2.8 |
| T0.5 | Feature streaming + `Rbar` / `kappa_hat` / trace-ratio / VCI fitting over ~290 checkpoints x 6 layers | 0.7 |
| T0.6 | **Gradient-masking pre-screen** on T0.3 models: unbounded-`eps`, `eps`-monotonicity (5 values), PGD-iteration plateau (10/20/50/100), transfer-vs-white-box, **Square-500 black-box**, {std, norm} x 3 seeds | 0.9 |
| T0.7 | **Encoder Lipschitz audit** (gradient-norm lower bound over 500 inputs + layerwise product upper bound) — moved up from Tier 2; **input to gate G2a in week 1** | 0.2 |
| | **Tier 0 total** | **9.9** |

**Gate G0 — spend Tier 1 only if all four hold:**
- (a) penultimate-layer `Rbar` on RN-18/CIFAR-10 train crosses **0.19** before end of training in >= 2/3 seeds, and exceeds the matched null by >= 10 null SDs;
- (b) per-class `kappa_hat` vs per-class test error gives Spearman `|rho| >= 0.5`, permutation `p < 0.05`, consistent in sign across 3 seeds, and **not** reproduced by the matched-null statistic;
- (c) the normalized-vs-standard difference in T0.3 **survives** T0.4's confirmation stage — i.e. at each arm's own selected LR, with 3 seeds per arm, the gap exceeds the stated MDE (~0.9 points clean, ~1.6–2.5 points robust) at `p < 0.05`;
- (d) the normalized arm **passes all five masking pre-screen diagnostics** in T0.6 at their numeric thresholds (unbounded `eps` <= 0.5%, `eps`-monotone, PGD-100 within 1.0 pt of PGD-50, transfer within 1.0 pt of white-box, Square-500 within 1.0 pt of white-box) in >= 2/3 seeds.

**STOP conditions:** if `Rbar` never leaves the null band, the atlas is a negative-result note (~3 pages) and the project ends at 9.9 h. If (c) fails — the gap vanishes or falls below the MDE once each arm is LR-tuned and re-seeded — **stop the robustness half of the project and publish that falsification**, which is a genuine contribution and costs 9.9 GPU-h to obtain. If (d) fails, **§6.6 is declared a gradient-masking artifact before any adversarial training is purchased**: T2.1–T2.3 are cancelled (18.2 h saved), and Tier 2 is re-scoped to a documented masking case study — T2.4/T2.5 on the 6 non-AT configs only (~4.8 h) — which is a publishable negative and is the outcome we consider most likely.

**Tier 1 — mechanism and detection. 41.2 GPU-h (cumulative 51.1).**

| # | Item | Est. |
|---|---|---|
| T1.1 | ResNet-18 CIFAR-100, {std, norm} x 3 seeds, 200 ep + superclass-pooled kappa analysis | 9.0 |
| T1.2 | ResNet-18 Tiny-ImageNet, {std, norm} x 1 seed (2nd seed only if G1 passes) | 4.0 |
| T1.3 | ViT-S/16 CIFAR-100, 150-ep DeiT-style recipe, {std, norm} x 1 seed | 7.0 |
| T1.4 | **`p`-sweep**: RN-18 CIFAR-10 with projection head `p in {64,128,256,512,1024}`, 2 seeds — the direct `kappa/p` test | 7.0 |
| T1.5 | Detection sweep: vMF likelihood vs MSP / ODIN / Mahalanobis / energy / arXiv:2505.12586, 4 attacks x 4 models, 1000-image subset | 5.0 |
| T1.6 | Adaptive attack vs the vMF detector: 100-step likelihood-constrained PGD, 5 `lambda`, 4 models | 3.0 |
| T1.7 | Matched-`A_p` injection: vMF vs renormalized Gaussian, 3 displacement levels x 2 families, train-time + test-time with EOT-20 | 6.0 |
| T1.8 | Batched per-row `(mu_i, kappa_i)` sampler + `S_outer` in-place fix (engineering; GPU only for tests) | 0.2 |
| | **Tier 1 total** | **41.2** |

**Gate G1 — spend Tier 2 only if at least one holds:**
- (e) vMF-likelihood AUROC >= Mahalanobis − 0.02 on >= 3/4 attacks **and** survives the adaptive attack at AUROC >= 0.65;
- (f) matched-`A_p` injection shows a vMF-vs-Gaussian robust-accuracy gap >= 2 points at some displacement level, outside the seed CI;
- (g) the `p`-sweep confirms the `Rbar > 0.19` boundary predicts kappa/null decoupling across `p`, **and** normalized arms show a systematically different `kappa_hat` trajectory at matched clean accuracy.

**STOP condition:** none of (e)–(g). Ship 6.2 + the `p`-sweep + the negative detection and injection results as a measurement/falsification paper at **51.1 GPU-h total**, and skip adversarial training and AutoAttack entirely.

**Tier 2 — robustness claims. 31.3 GPU-h (cumulative 82.4).**

| # | Item | Est. |
|---|---|---|
| T2.1 | Fast AT (FGSM-RS, cyclic LR, 50 ep) RN-18 CIFAR-10, {std, norm} x 3 seeds | 4.2 |
| T2.2 | Fast AT RN-18 CIFAR-100, {std, norm} x 3 seeds | 9.0 |
| T2.3 | **PGD-7 AT control**, CIFAR-10, {std, norm} x 1 seed — validates Fast AT, catches catastrophic overfitting | 5.0 |
| T2.4 | AutoAttack standard (APGD-CE, APGD-T, FAB-T, Square) on the **standard 1000-image CIFAR test subset**, 8 model configs @ ~0.8 h | 6.4 |
| T2.5 | Full obfuscated-gradient control suite (transfer, Square-5k, BPDA, unbounded-`eps`, `eps`-monotonicity, iteration-plateau, EOT-20, fp32/DLR) across the same 8 configs | 3.2 |
| T2.6b | 1-Lipschitz encoder (orthogonal conv + GroupSort, RN-18 scale, CIFAR-10, 1 seed) — only route on which a latent certificate composes non-vacuously; **trained only after G2b passes** | 1.5 |
| T2.6c | Restricted certified vMF smoothing: 200 imgs, `n0=100`, `N=1e4`, 3 `kappa` x {input-space, latent-on-Lipschitz} + matched Cohen Gaussian + matched-`A_p` Gaussian latent baseline — **runs only if G2a and G2b pass**, and is judged by G2c | 2.0 |
| | **Tier 2 total** | **31.3** |

(The former T2.6a Lipschitz audit is now T0.7, in Tier 0, so gate G2a resolves in week 1 rather than at the start of Tier 2. Total unchanged by the move; the 1.1 h increase over the previous draft is T0.4's confirmation seeds (+0.8) and T0.6's Square-500 (+0.3).)

**Summary.**

| Tier | Hours | Cumulative |
|---|---|---|
| Tier 0 (go/no-go, incl. masking pre-screen, LR confirmation, Lipschitz audit) | 9.9 | 9.9 |
| Tier 1 (mechanism, detection) | 41.2 | 51.1 |
| Tier 2 (robustness claims, incl. audited smoothing) | 31.3 | 82.4 |
| Slack / reruns / failed configs (20%) | 16.5 | 98.9 |
| **TOTAL** | | **~99 RTX-3090 GPU-hours** |

~4.1 days of pure GPU time on one card; realistically 3 weeks wall clock including analysis and the two engineering deliverables. Well inside the 1000-hour ceiling and inside the 100-hour target. **Under the STOP conditions the project terminates at 9.9 h, ~14.7 h (G0(d) masking failure path), or 51.1 h and still produces a paper.** Nothing in §6.10 is included in this total.

**What was cut to get from ~6,000 3090-hours to 99.** In descending order of cost removed:

1. **ImageNet-1k, entirely** (~2,100 3090-h of training alone), taking ResNet-50, ViT-B/16, and BREEDS ENTITY-13/LIVING-17 with it. This is the single biggest cut. It costs us the "large-scale atlas" framing; it does *not* cost us the feasibility finding, which now lands on CIFAR-100 and Tiny-ImageNet with a demonstrated fix.
2. **WRN-28-10 / WRN-70-16 adversarial training** (~900 3090-h) and the RobustBench-flagship comparison table. Replaced by ResNet-18-scale leaderboard reference points.
3. **PGD-10 Madry AT as the primary recipe and TRADES entirely** (~500 3090-h). Replaced by Fast AT + one PGD-7 control.
4. **Full-scale certified smoothing** (~250 3090-h): 500 images x `N=1e5` x 12 cells, plus SmoothAdv and denoised-smoothing baselines. Replaced by a 200-image / `N=1e4` probe behind three gates.
5. **The 30% blanket slack on a 2,400-hour base** (~1,700 3090-h), now 20% on an 82-hour base.

**What the adversarial-review audit changed inside the budget.** Added: the Tier-0 masking pre-screen (+0.6 h), its Square-500 black-box diagnostic (+0.3 h), the encoder Lipschitz audit moved into week 1 (+0.2 h), the 1-Lipschitz encoder that makes a latent certificate non-vacuous (+1.5 h), and the 3-seed LR-confirmation stage that closes the winner's-curse hole in gate G0(c) (+0.8 h) — 3.4 h of audit. Paid for in part by narrowing AutoAttack and the full masking suite from 10 model configs to 8 (−2.4 h), dropping the two single-seed Tiny-ImageNet arms, which carry no seed CI and therefore could never have supported a robustness claim. **Net +1.0 h, and no seeds were cut** — the reduction is in tasks, and the two most expensive additions (Square-500, LR confirmation) both *buy* statistical power, per the program's own rule that power is the thing it keeps getting wrong. Considered and rejected as offsets: trimming the adaptive-attack `lambda` grid from 5 to 3 (−1.2 h) and the C&W search levels from 5 to 3, both of which would have traded credibility for an hour we do not need to save at 99 < 100.

**What was explicitly *not* cut**, because it is cheap and load-bearing: the matched null everywhere, 3 seeds on all CIFAR arms *including at the tuned LR*, the per-arm LR sweep, the Tier-0 black-box screen, the full obfuscated-gradients control suite, the adaptive attack, and EOT-20 on every stochastic defense. Together these are ~15 GPU-h — 15% of the budget for essentially all of the section's credibility.

### 6.9 Risks and what a negative result looks like

- **6.2 — near-certain to yield something publishable, and it is now the 9.9-hour entry point.** Worst case: `kappa_hat` tracks the trace ratio at `rho > 0.95` and the difficulty correlation merely reproduces KappaFace. Still paper-grade as measurement methodology — a metric with a null, a boundary (`Rbar > 0.19`) validated by the `p`-sweep, and an honest statement of where NC1 is unmeasurable. Risk: judged incremental over VCI; mitigate by leading with the null, the boundary, and the `p`-sweep rather than the metric.
- **6.3.** Likely negative: the vMF likelihood ties Mahalanobis on AUROC and wins only on cost and on well-posedness where `n_class < p` — an efficiency/robustness-of-estimation result, weak alone, so it ships with 6.2. Real risk: the adaptive attack kills it, as adaptive attacks kill most latent detectors. Gate G1(e) is written so that this outcome stops the spend rather than absorbing it.
- **6.4.** Likely negative: at matched `A_p`, vMF and renormalized Gaussian are indistinguishable — a *useful* falsification showing the geometry argument is cosmetic once displacement is matched. If vMF wins it will be in the low-`kappa` (`A = 0.5`) corner; check there first, which the 3-level design does.
- **6.5 — highest risk, priced at 3.7 GPU-h behind three gates, two of which are free and one of which now resolves in week 1.** Three independent ways to fail: the composed input-space certificate is arithmetically vacuous on a standard encoder — G2a is expected to *fail* for that route, we learn this in week 1 for 0.2 h, and the standing publication rule pre-commits us to never printing a `1e-3` radius as a certificate; the derivation may not close (G2b, with a week-1 tripwire on the scalar inverter so we do not spend two weeks discovering an unstable quadrature); and even where it composes, arXiv:2108.00491 already occupies the latent-smoothing-on-Lipschitz-encoders setting, so the contribution reduces to the noise geometry and must clear G2c's paired MDE (~6 points certified accuracy, ~0.02 ACR) at matched `A_p`. Note the gate ordering is now strictly cheapest-first: 0.2 h of measurement, then 0 h of desk work, and only then the 1.5 h encoder and the 2.0 h smoothing run. The most likely honest outcome is: input-space vMF smoothing works and is a fair Cohen comparison; latent-space vMF smoothing is one paragraph reporting that geometry does not beat Gaussian at matched displacement.
- **6.6 — highest chance of being an artifact, and the audit made that assumption the default rather than the surprise.** Two independent ways to be wrong: gradient masking (base rate for "normalization confers robustness" surviving Square Attack and EOT is low) and the effective-LR confound. Both are now controlled *in Tier 0*, with numeric thresholds fixed in advance and a black-box attack in the screen rather than only in Tier 2, so the artifact outcome costs 9.9 h to establish rather than 27 h — and the day-3 tripwire on (d1)/(d4) starts the verdict arriving before Tier 0 even finishes. Expect inflated white-box robustness that collapses under Square-500 and/or an advantage that evaporates under per-arm LR tuning with 3 confirmation seeds. Publishable either way as a documented case study, **but only if the Tier-0 screen, the full checklist, and the LR sweep are pre-registered and reported regardless of outcome.**
- **Fast AT specifically.** Catastrophic overfitting would silently produce a "normalized is more robust" artifact if it hit one arm and not the other. Mitigated by per-epoch PGD-10 monitoring on a held-out slice, early stopping, and the PGD-7 control (T2.3) running *both* arms.
- **Cross-cutting:** the upward finite-sample bias aligns with every claim we want to make. Every reported `kappa_hat` carries its matched null in the same cell. **No null, no number.**

### 6.10 Deferred: needs language-model compute

Nothing in Tiers 0–2 requires a language model, and nothing was silently retained that does. Two extensions are genuinely language-shaped and are deferred. **Model policy for these items: anything trained or finetuned uses the Qwen3.5 family (Apache 2.0); anything purely *measured* deliberately keeps generational and architectural diversity, because the object of study is how representational geometry varies across training recipes and eras.**

**D1 — kappa atlas on transformer LM hidden states (small tier; ~9.5 GPU-h; unlockable inside this budget if Tier 0/1 justify it; not counted in the 98.9 above).** The atlas transfers directly to next-token-prediction models by conditioning on the target token instead of the class label: for a frequent token `v`, collect the residual-stream directions at the positions predicting `v` and fit `vMF(mu_v, kappa_v)`.

*Measurement substrates — diversity is deliberate and retained.* **Pythia-70m/160m/410m (`p = 512/768/1024`)** is the backbone of this item and is **not** substitutable: the suite ships ~150 public intermediate checkpoints per model, which is the only public *training trajectory* at this scale and therefore the only way to reproduce §6.2's onset-of-collapse curve in a language setting. No modern release replaces it. Alongside it we measure **GPT-2 small (`p=768`)**, **SmolLM2-135M/360M**, and **Qwen3.5-4B (`p=2560`)** as a modern-generation anchor — four generations and three tokenizer families, inference-only, no training, which is exactly the diversity the comparison needs.

*Feasibility arithmetic, recomputed for the wider modern model.* At `p=768` the 10p rule needs `n >= 7,680` occurrences per token — satisfied for the top ~2,000 BPE tokens in a 10M-token corpus slice, and *not* satisfied for the tail, which is the same admissibility boundary as §6.2 restated in a new domain. At Qwen3.5-4B's `p=2560` the rule needs `n >= 25,600`, so the same top-2,000 coverage requires a **~33M-token slice** (3.3x more streaming). That is cheap — extraction is one pass with `O(p)` memory — and it is itself the finding: *the admissibility boundary tightens linearly in model width, so the modern, wider models are exactly the ones where per-token kappa is hardest to estimate.*

*The from-scratch nanoGPT arm is cut.* Its only purpose was to obtain a training trajectory with checkpoints, and (i) Pythia already supplies ~150 of them for free, and (ii) under the model policy any from-scratch training would have to start at Qwen3.5-4B, which is not pretrainable at this budget by three orders of magnitude. In its place, the "does adaptation move kappa" question is answered by a **short Qwen3.5-4B LoRA SFT with ~10 intermediate adapter checkpoints** — LoRA freezes the base, so there are no base gradients or optimizer states and the bf16 footprint is ~10 GB, which fits the A6000 (48 GB) comfortably and the 3090 at short sequence length.

| D1 item | Est. |
|---|---|
| Pythia-70m/160m/410m: streaming `Rbar`/`kappa_hat` across ~150 public checkpoints x 3 sizes (no training) | 4.0 |
| Released-checkpoint diversity panel, inference-only: GPT-2 small, SmolLM2-135M/360M, Qwen3.5-4B over a 33M-token slice | 2.5 |
| Qwen3.5-4B LoRA SFT, ~20M tokens, 10 adapter checkpoints, A6000 (~2.5–3x a 1.5B model per token) | 3.0 |
| | **9.5** |

**Unlock condition (unchanged):** Gate G0 passes *and* the `p`-sweep (T1.4) confirms the `kappa/p` boundary predicts decoupling. **Not** unlocked merely because the vision atlas looks pretty.

**D2 — adversarial/jailbreak-input detection via vMF likelihood on an instruction-tuned LLM (OUT OF SCOPE on this hardware).** The natural robustness analogue of §6.3 is "does a vMF likelihood over residual-stream directions detect adversarial suffixes (GCG, arXiv:2307.15043) better than a linear probe?" Doing this at a scale anyone would believe means **Qwen3.5-9B**, with **Qwen3.5-4B** as the cheap pilot. VRAM is not the binding constraint it was in the previous draft: 9B LoRA in bf16 is ~20–24 GB and fits the A6000's 48 GB with room for the KV cache, and if a reference policy is needed it is free via adapter-disable (~18 GB saved at 9B). **Qwen3.5-27B dense (~56 GB LoRA) and the 35B MoE do not fit the A6000 at all** and would need the 2xA100 box, which is reserved for later scale demonstration — so 9B is the ceiling for this item and we say so rather than write 27B into a plan we cannot run.

*Recosted honestly.* The previous estimate of ~200–400 A100-80GB-hours assumed a 7B-class model. Qwen3.5-9B is roughly 1.2–1.4x a 7B per token, giving **~260–520 A100-80GB-hours** for the same design. Attack generation still dominates — GCG is ~1 GPU-hour per successful suffix at 500 steps, and a detection study needs several hundred suffixes across 3–4 model configurations — and GCG cost scales with the *target* model, so the 9B substitution moves the whole item, not just the probe. Running it instead as a 4B-only pilot would cost roughly **90–160 A6000-hours**, which is feasible on our own hardware but answers a weaker question (the point of D2 is that the directional score should win where covariance estimation breaks, i.e. at *large* `p`; 4B's `p=2560` is only 5x §6.3's `p=512`). Either way this needs a genuinely different machine or a genuinely longer calendar, not more time on the 3090.

**The result that would justify unlocking D2:** the vMF detector in T1.5/T1.6 must (i) match or beat Mahalanobis AUROC on >= 3/4 vision attacks, (ii) retain AUROC >= 0.65 **under the adaptive attack**, and (iii) show the advantage *growing* with feature dimension across the `p`-sweep in T1.4 — because the whole reason to move to an LLM is the claim that the directional score scales where the covariance-based score breaks. Absent (iii) in particular, D2 is not worth anyone's A100s and we should say so rather than write it into a grant.

**D3 — DoLoRA / per-component scales (belongs to the adapter project, flagged here to prevent scope leakage).** The sharp question — *does a per-component scale vector `s_i` beat DeLoRA's single global scalar?* (DeLoRA, arXiv:2503.18225, ICLR'25, already merged into HuggingFace PEFT; DoLoRA's own Appendix A proves a global scale is a strictly worse approximator under non-uniform target singular values) — is **architecture-agnostic and does not need a large model.** It should be answered in this program on ViT-B/16 VTAB-1k (19 tasks x 1000 images; ~1–3 GPU-h per method+seed, the best credibility-per-GPU-hour setting available) and, if a language check is wanted, on **Qwen3.5-4B LoRA SFT** on the A6000. The previous draft costed that language check at ~1–3 GPU-h against a 0.5B model; 4B is the smallest model the policy permits and is ~8x a 0.5B per token, so holding the per-cell cost near **~3 GPU-h per method+seed** requires cutting scope: the SFT token budget drops from 50M to 15M and the rank grid from `{4,8,16,32,64}` to `{8,32}`. **Seed count is unchanged** — ranks and steps were cut first, deliberately, since power is the thing this program keeps under-provisioning. None of that lives in Project 6, and no part of Project 6's budget is reserved for it.


---

## Cross-cutting: shared infrastructure, sequencing, and portfolio management

> **Numbering note — read before this section.** This section was drafted independently and uses its
> own project numbers, which do **not** match the rest of the document. It also merges two projects
> and proposes one that has no section of its own. Translation table:
>
> | Label used *in this section* | Canonical project | Note |
> | --- | --- | --- |
> | P1 | **P1** continual learning | same |
> | P2 | **P4** + **P5** | merges the geometry and generative projects into one paper |
> | P3 | *(no section)* | a **new** proposal: nViT/nCNN/nTransformer with honest token-vs-wall-clock accounting. See below — it is worth keeping as a candidate |
> | P4 | **P2** DoLoRA | |
> | P5 | **P3** RL on code | |
> | — | **P6** robustness | added after this section was drafted; not costed here |
>
> Two of this section's independent judgments are worth taking seriously precisely because they were
> reached without coordination: that the geometry and generative projects are really one paper, and
> that a *negative* result on normalized vision architectures ("the token-efficiency gain largely
> vanishes against a properly-tuned baseline, and is net-negative in wall-clock") would be a
> valuable, citable contribution. Its compute figures are also the ones the review in section 8
> finds most credible, and they conflict with the per-section budgets by 3–16×.

| # | Short name | Primary venue |
|---|---|---|
| P1 | Spherical representations for class-incremental continual learning (nMLP/nCNN, CIFAR → ImageNet-scale) | ICLR |
| P2 | Null-calibrated directional statistics for representation geometry + high-dimensional vMF latents (hyperspherical VAE past m=40) | ICLR |
| P3 | Normalized architectures beyond nGPT: nViT / nCNN / nTransformer with honest token-vs-wall-clock accounting | CVPR |
| P4 | Geometry of low-rank adapters: per-component strength, tangent-space retraction, and κ̂-measured adapter direction distributions; vision PEFT (VTAB-1k) | CVPR |
| P5 | Normalized low-rank adapters for RLVR on code generation | ICLR |

---

## Shared infrastructure

### Engineering deliverables that multiple projects depend on

Estimates assume one competent PyTorch engineer at 1.0 FTE; "eng-weeks" includes tests and docs, not paper-writing. Hardware tier for all compute figures: a shared cluster with intermittent access to one 8×A100-80GB node, occasional 8×H100, plus a 64-core CPU node and one M3 Max for the CPU-only statistics work.

| ID | Deliverable | Eng-weeks | Blocks | Notes |
|---|---|---|---|---|
| (a) | Batched per-row (μ_i, κ_i) vMF sampling | 2–3 | P2 (hard), P4/P5 (soft) | New work: the current `VonMisesFisherTorchHH` class binds one (μ, κ) per instance |
| (b) | Stable differentiable vMF KL / Bessel-ratio autograd at large m | 2 + 1 for tests | P2 (hard) | Nothing else in the program needs gradients through κ |
| (c) | Normalized-layer library (nMLP, nCNN, nViT, nTransformer) | 4–6 | P1, P3 (hard), P4 (soft) | Critical path; one implementation, not two |
| (d) | PEFT-compatible normalized adapter module | 2–3 + 1 integration | P4, P5 (hard) | Must be a `peft` tuner subclass, not a fork |
| (e) | κ null-calibration tables + drift-measurement utility | 3 | P1, P2, P4 (hard) | ~200–400 CPU-hours, embarrassingly parallel; no GPU |
| (f) | Experiment tracking, seed management, results DB | 1–2 | everything | Cheapest, do it in week 1 |

**(a) Batched per-row (μ_i, κ_i) vMF sampling.** The repo's samplers fix μ and κ at construction and amortize the Householder reflection across draws. P2 needs a *different* (μ_i, κ_i) per row of a minibatch. Two pieces of work: (i) vectorize Ulrich's rejection loop with an active mask and per-row acceptance thresholds, resampling only unaccepted rows until the mask empties (the hyperspherical-VAE paper's acceptance rate of 1.0–1.46 and its *improvement* with dimension means the loop terminates in 1–2 iterations at m ≥ 64; budget the tail anyway); (ii) batched Householder — the fused `addmm` rank-one update generalizes to a per-row reflection vector as a `baddbmm`, but the in-place variant must be re-audited (see the not-truly-in-place `S_outer` bug already found in this repo). Gradient support via implicit reparameterization of the rejection sampler (Naesseth et al., 2017) is a further ~1 week and only P2 needs it. Deliver `sample_batched(mu: (B, p), kappa: (B,), n_per_row: int)` with a fp32/bf16/fp64 parity test against the existing scalar path.

**(b) Numerically stable differentiable vMF KL / Bessel ratio at large m.** KL(vMF(μ,κ) ‖ Unif(S^{m−1})) = κ·A_m(κ) + log C_m(κ) + log(surface area), with A_m(κ) = I_{m/2}(κ)/I_{m/2−1}(κ). The forward pass must route through compiled CUSF or the pure-Torch `torch_log_iv` port — `scipy.special.ive` underflows to exactly zero above order ~511, so any autograd path built on SciPy returns NaN at m ≥ 1024, which is precisely the regime P2 exists to reach. Register a custom `autograd.Function` with the analytic derivative A′(κ) = 1 − A(κ)² − ((m−1)/κ)·A(κ) rather than differentiating through the log-Bessel series; the series is accurate to ~1e-13 in value but its autograd graph is deep and its backward is not obviously stable. Test: gradcheck in fp64 at m ∈ {8, 40, 128, 1024, 4096} and κ/m ∈ {0.01, 0.2, 1, 10}, plus a finite-difference cross-check. CI must fail if the SciPy path is selected at m > 511.

**(c) Reusable normalized-layer library.** P1 and P3 must share one implementation or the program produces two divergent "nMLP"s and the continual-learning result cannot be inherited by the vision paper. Scope: `NormLinear`, `NormConv2d`, `NormAttention`, `NormMLPBlock`, and the nGPT residual update h ← Norm(h + α ⊙ (Norm(BLOCK(h)) − h)) with learnable per-coordinate eigen-learning-rates α; unit-norm weight constraint applied as a post-optimizer-step retraction (a hook, so it composes with any optimizer); a switch for *representation-only*, *weight-only*, and *both* normalization so the mechanism ablation (below) is a config flag, not a rewrite. Include a fused normalization path (`torch.compile` first, hand-written Triton only if the compiler leaves >20% on the table) — nGPT's own 60–80% per-step wall-clock penalty from unfused normalizations is the single most attackable weakness in the whole normalized-architecture story, and P3's headline honesty depends on measuring it, not inheriting it. Also required: a *strong* baseline mode matching modded-nanogpt's QK-Norm + zero-init projections, so P3 never compares nGPT to vanilla nanoGPT.

**(d) PEFT-compatible adapter module.** Implement as a `peft.tuners.lora`-style tuner subclass so it drops into TRL's `GRPOTrainer` and verl's FSDP/PEFT path unchanged. Required modes in one class: plain LoRA (α/r), rsLoRA (α/√r), DoRA with the published *detached* column norm **and** an exact-gradient variant, DeLoRA (per-rank-one normalization, single learnable scalar λ per layer, ‖W‖ folded in), and the per-component variant (S = diag(s_i), s_i = softplus(λ_i) − log 2, λ_i = 0 at init). The DeLoRA parity mode is not optional: it is the head-to-head baseline, it is already merged into HuggingFace PEFT, and any comparison that does not control for DeLoRA's ‖W‖ folding is a confound reviewers will find. Ship a `to_lora_pair()` export (B = Û S, A = V̂ᵀ) so rollout-time inference uses vLLM's native `lora_request` with zero custom kernels — this makes P5's inference path identical to the baseline's, which removes an entire class of "is the speedup real?" objection.

**(e) κ null-calibration tables and drift-measurement utility.** Two artifacts.

*Null tables.* For a grid p ∈ {2, 4, 8, …, 8192} × n ∈ {10², 10³, 10⁴, 10⁵} × estimator ∈ {Banerjee, Sra, converged Newton}, draw 1000 replicates of n uniform unit vectors in R^p, record R̄ and κ̂, and store mean, median, and the 2.5/50/97.5 percentiles in a Parquet table with a log-log interpolation API `null_kappa(p, n, estimator) -> (mean, lo, hi)`. Only R̄ = ‖mean vector‖ needs to be materialized, so the sampler streams in chunks and never allocates the full n×p matrix. Cost ~200–400 CPU-hours, one to two days on a 64-core node. This table is what makes every κ̂ in every paper interpretable, and it is the cheapest credibility purchase in the program.

*Drift utility.* `measure_drift(ckpt_a, ckpt_b, probe_set, layer)` → normalize representations, compute the per-sample angular displacement direction (h_b/‖h_b‖ − h_a/‖h_a‖, renormalized), fit vMF, and report κ̂ against the matched null. Alongside it — never instead of it — CKA (linear and RBF), orthogonal-Procrustes disparity on the normalized representations, principal angles between top-k subspaces, and RSA correlation. The utility must return a hard feasibility flag: if κ̂/p < 0.05, or n < 10p, the estimate is returned but marked `underdetermined=True`, and the plotting layer renders it greyed with the null band overlaid. This is the concrete mechanism that turns the existing 3D-PCA eyeball into a measurement, and it is the methodological contribution the program is actually selling.

**(f) Experiment tracking, seeds, results DB.** W&B (or MLflow) for curves; a DuckDB `results` table as the source of truth for every number that reaches a paper. Canonical row schema: `run_id, git_sha, config_hash, seed, hardware, wall_clock_s, step_count, token_count, metric_name, metric_value, eval_harness_commit, eval_task_version, metric_convention (acc | acc_norm | generative)`. Seed management: a single `set_all_seeds(s)` covering Python/NumPy/Torch/CUDA plus dataloader worker seeding, and a rule that seeds are 1..5 and never cherry-picked. Papers pull tables directly from DuckDB via a script checked into the repo; no hand-transcribed numbers.

### The shared measurement layer

Every project emits the same `geometry_report_v1` JSON blob, one record per (run, checkpoint, layer, probe-set), written into the results DB. Papers select from it; nobody re-implements a diagnostic. The common set:

1. **Mean resultant length R̄** of the normalized representation directions — reported raw. R̄ is the honest primitive; κ̂ is a violently nonlinear transform of it and is unstable at low concentration.
2. **Null-calibrated κ̂** of representation directions, pooled and per-class, always as (κ̂, null mean, null 95% band, κ̂/p, feasibility flag). Never a bare κ̂. The finite-sample bias is systematically upward and points the same direction as any "more concentrated than uniform" claim, so the matched null is mandatory, not decorative.
3. **Null-calibrated κ̂ of the angular-drift distribution** between two checkpoints on a fixed probe set (P1's central measurement, but reported by all).
4. **Classical drift metrics**: linear + RBF CKA, Procrustes disparity, top-k principal angles, RSA. Reported alongside κ̂ so that a κ̂ finding that no classical metric corroborates is visible as such.
5. **Pairwise cosine-similarity distribution**: mean, sd, 1/50/99th percentiles, against the uniform reference (mean 0, sd ≈ 1/√p). This is nGPT's own ad-hoc diagnostic; keeping it lets the program's papers be compared to nGPT's directly.
6. **Spectral shape of the representation Gram matrix**: effective rank (exponential of eigenvalue-spectrum entropy) and stable rank.
7. **Weight geometry**: per-row and per-column norm distributions, condition number, and — for normalized layers — the residual off-tangent component of the raw update before retraction.
8. **Relative update magnitude** ‖Δw‖/‖w‖ per layer per step, logged every step. This is simultaneously a geometry diagnostic and the effective-learning-rate control (see the last section). Non-negotiable.
9. **Optimization health**: token-level entropy (LM/RL runs), logit margin (classifiers), gradient norm.
10. **Both clocks**: step/token count *and* wall-clock, on every curve, in every paper.

---

## Sequencing

**Cheap and derisked (do first).** Two experiments in the first six weeks decide the shape of the program:

- **D1 — the rsLoRA kill-shot.** Gemma-3-4B, commonsense170k, r=2, three seeds, four arms: LoRA at the canonical LR, LoRA with a per-rank LR sweep, rsLoRA (α/√r), DeLoRA. ~12 runs × 14–20 A100-h ≈ **200–250 A100-hours**. If a properly scaled and LR-tuned LoRA baseline closes most of the reported +7.63 at r=2, the entire method framing of P4/P5 collapses to a diagnostics framing, and it is far better to learn that in September 2026 than in April 2027. Run this before writing a line of P4.
- **D2 — the continual-learning replication.** CIFAR-10, 5+5 class-incremental, MLP and nMLP, 5 seeds, plus ResNet-18/nCNN. ~1.5–2 GPU-h per run, ~30–40 A100-hours total. Validates (c), (e), and (f) end-to-end on a workload small enough to iterate hourly, and produces the first null-calibrated drift number in the program.

**Critical path.** (f) → (e) → (c) → **P1** → **P3**. The normalized-layer library is the long pole: P3's expensive vision runs cannot start until it is fused and correct, and re-running DeiT-scale training because a normalization was in the wrong place costs ~420 GPU-hours per mistake. A parallel path runs (d) → **P4** → **P5**; P5 must not start until P4's adapter has passed a numerical-parity test against HuggingFace DeLoRA, because debugging an adapter inside a GRPO loop where 70–85% of wall-clock is rollout generation is miserable.

**Parallelism.** P1 and P4 are independent after week 6 and should run concurrently on different people. P2 depends only on (a) and (b) and touches nothing else — it is the natural background/solo project, and the right thing to hand to a student who wants a self-contained problem. P3 and P5 are both gated: P3 on P1 showing a real effect worth scaling, P5 on P4 surviving D1.

**Expensive and speculative.** P3 is the most expensive line item in vision (DeiT-S ≈ 240 A100-h/run, DeiT-B ≈ 420, ResNet-50/ImageNet ≈ 67; a credible nViT study is 15–25 runs ≈ **4,000–7,000 A100-hours**) and the most exposed to the strong-baseline problem. P5 at 7B is the most expensive overall (a rank sweep at 1.5B ≈ 2.5k H100-h; 7B endpoint confirmation ≈ 2.5k more). The full LLM PEFT grid described in the brief (3,500–4,500 A100-h) must be cut to one model and ranks {2, 8, 32} unless funding appears. VTAB-1k is the outlier in the other direction: 19 datasets × 1k images ≈ **10–20 A100-hours per method-seed**, the best credibility-per-GPU-hour in the entire program, and the natural home for the vision extension that DoRA/DeLoRA explicitly list as future work.

**Calendar (team ≈ 2.1 FTE: Andreas at ~0.6 research FTE, one engineer at 1.0, one student at ~0.5).** Deadlines are referred to by their typical windows, not by dates.

- **Aug–mid-Sept 2026.** (f) week 1; (e) weeks 1–3; (c) started; D1 and D2 running by week 3. The ICLR window that falls in the *typical Sept/Oct* slot this year is roughly seven weeks out and is **not reachable** — the only candidate would be a rushed P1, and a continual-learning paper submitted without the effective-learning-rate control fully executed will be read as an artifact and rejected. Do not try.
- **Mid-Sept–Oct 2026.** (c) and (d) complete. P4's VTAB-1k sweep (full grid vs. full FT, linear probe, BitFit, VPT, AdaptFormer, SSF, LoRA, rsLoRA, DoRA, DeLoRA, FacT, ALoRE) plus FGVC and few-shot. P1 scaled from CIFAR-10 to CIFAR-100 and ImageNet-subset class-incremental, with the geometry report wired in.
- **Typical Nov 2026 CVPR window.** Submit **P4** (vision PEFT + null-calibrated adapter geometry). Post to arXiv the same week — with DeLoRA already in PEFT, priority on the *diagnostic* framing is the only priority available, and it is worth defending early. P1 is a stretch second submission here if it is vision-framed; otherwise hold it.
- **Dec 2026–Mar 2027.** (a) and (b) → P2 built and run (cheap: hyperspherical VAEs at m up to 4096 on MNIST/CIFAR/CelebA are single-GPU-days, not weeks). P5 adapter RL at Qwen2.5-Coder-1.5B, full rank sweep. P1 finished properly, including the ELR protocol in full.
- **Apr–Aug 2027.** P3's expensive runs (nViT on ImageNet-1k, nGPT vs. modded-nanogpt at matched wall-clock). P5's 7B endpoint confirmation (r ∈ {2, 4}, 6 runs).
- **Typical Sept/Oct 2027 ICLR window.** Submit **P1**, **P2**, **P5**.
- **Typical Nov 2027 CVPR window.** Submit **P3**.

That is four papers submitted within ~15 months and five within ~16, which is aggressive but not fantastical for 2.1 FTE *given* that the measurement layer is built once. If (c) slips past October 2026, drop P3 to the following cycle rather than compressing it — it is the project where undercooking is most expensive and least recoverable.

---

## Portfolio view

| Project | Venue | Novelty risk | Execution risk | Compute (A100/H100-h) | Depends on | Negative result publishable? |
|---|---|---|---|---|---|---|
| **P1** Continual learning with spherical representations | ICLR | **Medium.** Normalization-helps-forgetting is plausible-adjacent to existing work; the *null-calibrated drift measurement* is the novel part, not the finding | **Medium.** Original code not retained; the whole result is currently an eyeball on a PCA plot | ~300–600 (CIFAR + ImageNet-subset, with LR sweeps) | (c), (e), (f) | **Yes, strongly.** "nMLP's apparent stability is an effective-learning-rate artifact, and here is the protocol that shows it" is a genuinely useful ICLR paper |
| **P2** Directional statistics for representations + high-m vMF latents | ICLR | **Low-medium.** Nobody has null-calibrated κ̂ on embeddings; pushing hyperspherical VAEs past m=40 is a clean, unclaimed gap | **Low.** Self-contained, mostly the researcher's home turf, small compute | ~100–300 | (a), (b), (e) | **Yes.** "vMF latents do not scale past m≈X and here is the precise statistical reason (κ/p, not surface area)" is a better paper than a mild win |
| **P3** Normalized architectures in vision / honest wall-clock | CVPR | **Medium-high.** nGPT exists; nViT is an extension, and the honest-accounting angle is the real contribution | **High.** Strong baselines (QK-Norm, zero-init) already absorb much of the claimed benefit; DeiT-scale reruns are punishing | **4,000–7,000** | (c), (f), P1's result | **Yes, and it may be the best version.** "Normalization's token-efficiency gain is largely absent against a properly-tuned modern baseline, and net-negative in wall-clock" is a valuable, citable negative |
| **P4** Adapter geometry + vision PEFT | CVPR | **High.** DeLoRA (ICLR 2025, arXiv 2503.18225, merged into HF PEFT) already owns per-rank-one normalization, exact gradients, and the tangent-space projection. Only per-*component* strength and the κ̂ diagnostics are left | **Medium.** VTAB-1k is cheap and well-trodden; the risk is scientific, not logistical | ~400–800 (VTAB-1k grid + FGVC + LLM ablations) | (d), (e), (f), D1 | **Partially.** "Per-component strength ≈ DeLoRA's scalar" is an ablation, not a paper — it is only publishable if the geometry diagnostics carry it. Plan for that from day one |
| **P5** Normalized adapters for RLVR on code | ICLR | **Medium.** The contested r∈{1,2,4} regime (LoRA-Without-Regret vs. verl's r≥32 guidance) is genuinely open | **High.** RLVR is noisy at 1–2 points, reward hacking is real, sandbox/executor throughput is its own engineering project | **2,500 (1.5B sweep) + 2,500 (7B endpoints)** | (d), (f), P4 validated | **Yes.** "RLVR is rank-insensitive down to r=1 in a second domain and a second adapter family" is publishable *if* the rank sweep, ≥3 seeds, pass@k, and entropy logging are in place from step one |

Blunt reading of the table: **P2 and P4-as-diagnostics have the best risk-adjusted return**; P4-as-method is dead on arrival; **P3 is the one to cut** if the budget or the calendar tightens, because it is simultaneously the most expensive, the most crowded, and the least dependent on anything the repo uniquely provides.

---

## The unifying narrative

**The honest answer: this is roughly two-and-a-half papers of one program plus two-and-a-half adjacent papers that happen to share a codebase.**

The thread that genuinely exists is: *constraining representations and parameters to the unit sphere changes optimization geometry in ways that are measurable with directional statistics and that plausibly help continual, low-rank, and sparse-signal learning.* Three claims are bundled there — that the constraint changes geometry, that the change is measurable with vMF machinery, and that the change is beneficial. Only the middle claim is uniquely the researcher's, and it is the one the repo actually supports.

Against that thread:

- **P1 genuinely supports it.** Spherical constraint → representations move less during phase 2 → less forgetting, with the movement measured as a null-calibrated κ̂ of the angular drift. Mechanism, measurement, and benefit are all in one story.
- **P4 genuinely supports it** — but only in its diagnostic form. The `(I − x xᵀ)` factor in the DeLoRA/DoLoRA gradient *is* the sphere's tangent-space projection, and asking "what distribution do the adapter directions actually settle into, and is it distinguishable from uniform?" is a question no PEFT paper has asked and that this repo is uniquely equipped to answer. The method contribution is gone; the geometry contribution is intact.
- **P3 supports it at arm's length.** nGPT already made the architectural argument. P3's contribution is measurement discipline and honest accounting — valuable, but it is auditing someone else's claim rather than advancing the thread.
- **P2 is adjacent.** Its subject is directional statistics, so it *looks* central, but the mechanism is a generative-model likelihood, not optimization geometry. A vMF posterior with learned κ is not a constrained optimizer. It shares the toolkit, not the thesis.
- **P5 is adjacent.** The interesting question in RLVR-with-low-rank-adapters is bits-per-episode and rank sensitivity. Sphericity rides along; if the adapter geometry turned out to be irrelevant, the paper would be unchanged in its most interesting parts.

Why this matters for time allocation and not for reviewing: no reviewer of any single paper will care about the program, and none of these papers should open by gesturing at the other four. But the *researcher* should notice that the shared measurement layer — item (e), plus the `geometry_report_v1` schema — is the only asset that appears in every project, and that it is also the thing nobody else has. If forced to choose one thing to make excellent, make it that. A well-executed measurement layer turns P1 and P4 into papers with a defensible contribution even when their headline effects shrink; without it, both are ordinary papers competing with better-resourced groups on effect size alone.

---

## Common failure modes across the program

### 1. The effective-learning-rate confound (threatens P1, P2, P3 simultaneously)

Normalizing weights to unit norm makes gradient descent scale-invariant: the effective step size in the tangent direction scales as η/‖w‖², so an "architecture" comparison between a normalized and an unnormalized network is silently also a learning-rate-schedule comparison (van Laarhoven 2017; Hoffer et al., "Norm matters", NeurIPS 2018; Arora et al., "Theoretical analysis of auto rate-tuning", ICLR 2019; Li & Arora, exponential LR schedules, ICLR 2020). nGPT's learnable α settling at 0.20–0.33 is itself a learned step-size. This single confound can explain "converges faster on new classes", "forgets less", and "4×/10×/20× token efficiency" without any of them being about the sphere.

**One protocol, implemented once in the shared library as a training callback, cited from a single shared appendix in every paper. Call it the ELR protocol.**

- **Tier 0 — no shared learning rates, ever.** Every arm gets its own log-grid LR sweep with an *equal trial budget* (≥7 points, ≥1 octave beyond the best point on each side), and the paper reports the full sweep curve, not just the best point. A comparison at a single shared LR is not admissible anywhere in the program. This also discharges the "Learning Rate Matters: Vanilla LoRA May Suffice" (2602.04998) objection for P4/P5.
- **Tier 1 — measure and match the relative update.** Log ‖Δw‖/‖w‖ per layer per step for every arm (diagnostic #8 above) and plot its trajectory. Then run a *norm-matched control*: the unnormalized network with per-layer LR rescaled so its median ‖Δw‖/‖w‖ tracks the normalized arm's within 10% across training. If the normalized arm's advantage survives against the norm-matched control, the claim is about geometry; if it does not, the claim is about step size, and the paper says so.
- **Tier 2 — decompose the mechanism.** Five arms, all config flags in (c): (i) weight normalization only; (ii) representation normalization only; (iii) both, i.e. the full nGPT block; (iv) unnormalized + LayerNorm/RMSNorm; (v) unnormalized + a scale-invariant optimizer (LARS/LAMB) as the "effective-LR-controlled" reference. Run with SGD as well as AdamW — Adam's per-parameter normalization already partially launders the effect, and using Adam alone hides the mechanism.
- **Tier 3 — both clocks, always.** Steps/tokens *and* wall-clock, with per-step normalization overhead reported separately from the algorithmic gain, so a token-efficiency win that is a wall-clock loss is legible as such.

For P4/P5 the same protocol takes an adapter-specific form: the α/r vs α/√r scaling is an effective-LR knob, LoRA's optimal LR is ~rank-independent only *because* of the 1/r scaling, and "gains are largest at low rank" is exactly the signature a mis-scaled baseline produces. D1 is Tier 0 of the ELR protocol applied to adapters, which is why it runs first.

### 2. Reporting κ̂ that the data cannot support

Statistical error dominates numerical error by ~12 orders of magnitude, the finite-sample bias is systematically upward, and it points the same way as any claim of concentration. Enforced in code, not in prose: no API returns a bare κ̂; every return carries the matched-null band and a feasibility flag (κ̂/p ≥ 0.05 and n ≥ 10p, tightening to n ≥ 25p for a 1% claim). Papers reporting a flagged estimate must show the null band on the same axis.

### 3. Bessel underflow above p = 511

`scipy.special.ive` is exactly zero above order ~511, so SciPy log-Bessel is −inf, Sra-with-SciPy is NaN at p ≥ 1024, and safeguarded Newton stalls at ~1e-3 relative error without complaining. Every project touching p ≥ 1024 — which is every project with a modern backbone — must use compiled CUSF or the Torch port. Add a CI test that fails the build if the SciPy path is reachable above order 511, and pin the CUSF build in the environment lockfile.

### 4. Undertuned baselines

Beyond LoRA's LR: DoRA must be run in its *published detached-norm* form (or both forms, labelled), DeLoRA must be run with its ‖W‖ folding intact, nGPT must be compared to a modded-nanogpt-class baseline with QK-Norm and zero-init rather than vanilla nanoGPT, and continual-learning baselines must include tuned EWC/LwF/rehearsal-free SOTA rather than a naive fine-tune. A reviewer's first hypothesis for any positive result in this program will be "the baseline was weak", and in three of the five projects that hypothesis is a priori reasonable.

### 5. Seed and statistics discipline

≥3 seeds everywhere, 5 in the low-rank and RLVR regimes where per-seed std runs 1.5–1.9 points. Mean ± std in the main table, per-seed numbers in the appendix, paired Wilcoxon signed-rank across tasks, explicit statement that hyperparameters were tuned equally across arms, and never best-of-run. A +1.0 average with ±0.8 std over 3 seeds is not a result and should not be written as one.

### 6. Evaluation-protocol drift

Pin the `lm-evaluation-harness` commit and report per-task `version`; state `acc` vs `acc_norm` per task in a footnote (HellaSwag/ARC-c/OBQA conventionally `acc_norm`; BoolQ/PIQA/WinoGrande `acc`); state whether scoring is generative (LLM-Adapters protocol) or log-likelihood (harness) and report the other in the appendix. Use the canonical **8-task** commonsense170k suite, not a 5-task subset. For code: LiveCodeBench with the time window quoted explicitly, pass@1 with the n=16 unbiased estimator *and* pass@k up to 64, because RLVR is known to raise pass@1 while lowering pass@k.

### 7. Reward hacking and contamination (P5)

Read-only sandbox, subprocess isolation, AST checks banning test-module imports, and held-out tests never seen during RL (train on 60% of tests, score on 100%). Log token-level entropy every step and publish the plot; entropy collapse within a few hundred steps is the default outcome, not an anomaly.

### 8. Non-retained code as a program-level risk

Two of the program's three seed results (the continual-learning finding and the nLoRA/DoLoRA experiments) have no surviving code. Treat replication as a gated deliverable with an explicit go/no-go, not as a warm-up: D2 gates P1, D1 gates P4 and P5. If a replication fails to reproduce the qualitative direction within its budget, the correct move is to reframe toward the diagnostic contribution immediately rather than to keep spending on the effect.


---

## Compute tiers and the escalation ladder

### Where the program landed

All figures are single-A6000 hours unless noted. See the envelope section for hardware rules.

| Project | Tier 0 | Full | Venue | What Tier 0 decides |
| --- | --- | --- | --- | --- |
| **P1** continual | **10.0** | 400 (~1,480 runs) | CVPR main track | Does the nMLP advantage survive a tuned baseline (≥3.0-pt median paired gain, CI excluding 0) **and** dominate the stability–plasticity frontier on ≥60% of the epoch grid? |
| **P2** adapters | **7.9** | 571 (100 = the vision paper) | ICLR | Does per-component scaling beat a global scalar above the seed floor at low rank? |
| **P3** RL on code | **6.8** | 656 (~186 expected after gates) | ICLR | Is the objective's information content a real axis at all? A failed Gate 0 unlocks no LLM compute, ever |
| **P4** geometry | **4.0** | 72 + 350 CPU-core-h | ICLR | Does attention-head κ̂ separate induction heads at AUC ≥ 0.70? |
| **P5** VAE | **5.0** | 97 | ICLR | Finite vMF-KL gradients at m ∈ {1024, 4096}, and MNIST reproduction within 2 s.d.? |
| **P6** robustness | **8.0** | 99 (stops at 8 or 49 under STOP gates) | see §11 | Does R̄ ever leave the null band, and does the gap survive per-arm LR tuning? |

**Whole program: ~1,900 GPU-hours (the review argues 1,900–1,980 across two card types). Every gate in it opens for ≈42 GPU-hours.**

That second number is the one to act on. Roughly two days of compute decides which of six projects
are real, and the expected spend is far below the committed total because most gates are designed to
fail. For comparison, the first draft of this program budgeted ~20,900 GPU-hours and contained a
single project that could not reject its own null at any price.

### Statistical power, since that was the original failure

The review's most damaging finding was that P3's interaction test had a minimum detectable effect of
5.5 points at 3 seeds against a predicted effect that might be 2.5. The first attempt to fix that
bought 12 seeds — far outside normal practice, and the wrong instrument. The design now runs **3–5
seeds**, standard for the field, and recovers resolution from replication that is cheaper than seeds:
common-random-number pairing across arms, pooling across ranks, and a per-problem mixed-effects
analysis over MBPP-Plus's 378 problems instead of one run-level mean. Where that still is not enough,
the estimand is demoted rather than the seed count inflated:

| Claim | Design | MDE |
| --- | --- | --- |
| P2 vision `gain(r=2)` | VTAB-19 × 3 seeds, ViT-S/16, paired Wilcoxon | **0.58** VTAB-mean pts |
| P2 language 8B replication | 2 seeds, 8 tasks | **1.30** pts — a 5.9σ margin against the claimed +7.63 |
| P3 primary (**simple effect**, after demotion) | pooled r ∈ {1,2}, 4B, **5 seeds**, CRN pairing, 378-problem mixed effects | **1.4** pts |
| P3 interaction (**secondary, underpowered**) | same design | 1.6–2.2 pts vs a modal effect of +1.5…+4 — reported with CI and MDE, gates nothing |
| P3 9B tier | 3 seeds | 1.7 pts — a real test, not just a consistency check |

P2's language tier makes the right call explicitly: going from 2 to 5 seeds at 8B only improves the
MDE from 1.30 to 1.06, because the task-heterogeneity term dominates. Ninety extra hours buy 0.24
points, so they are not spent.

P3's honest caveat is pre-registered: the modal predicted interaction is +1.5 to +4, so the bottom
third of that range yields a bounded null, and the write-up commits in advance to reporting it as
one.

---

### Tier B — the earned pool

A few thousand H100-hours are available for high-belief projects, plus 2×A100 for scale
demonstration. Both are **earned, not allocated**, under one rule:

> **Scale buys confirmation, not discovery.**

Establish the effect at the default tier with enough seeds that the CI excludes zero; confirm it has
the predicted shape (for this program, almost always that it grows as the learning signal gets
scarcer); only then escalate. A scaled run returning "we cannot tell" is a total loss. The same
outcome cheaply is often informative.

| Priority | Project | Ask | H100-h | Unlocked by |
| --- | --- | --- | --- | --- |
| 1 | **P3** | Long-horizon RLVR: 2048-token completions, G=16, full LiveCodeBench-v6, 9B | ~800 | Gate 2 **plus** a positive short-completion interaction |
| 2 | **P1** | Full-fidelity 88-epoch ImageNet CIL, for absolute-accuracy claims | ~340 | Frontier dominance holding at ResNet-18 scale |
| 3 | **P2** | Powered ≥5-seed × 4-rank per-component vs global at 8B | ~250 | The 8B point estimate landing in the ambiguous 0.3–1.3 zone |
| 4 | **P3** | Second model family (Olmo-3-7B) Tier 2 replication | ~220 | A confirmed 4B → 9B effect |
| 5 | **P6** | Jailbreak / adversarial-suffix detection on ≥1B instruction-tuned LLMs | 150–250 | vMF detector matching Mahalanobis, surviving adaptive attack at AUROC ≥ 0.65, advantage growing with p |
| 6 | **P1** | nViT-full with LayerNorm removed, ViT-S + ViT-B re-adaptation | ~150 | Frontier dominance transferring to attention architectures |
| 7 | **P4 / P5** | nGPT at p ≥ 1024; Optimus-style hyperspherical text VAE | 60–350 each | Project-specific gates |

Everything passing exceeds "a few thousand," which is the point of the ordering. Two items are
**permanently out** regardless of results: from-scratch normalized ViT/DeiT on ImageNet-1k (~1,500
H100-h) and any QLoRA-based adapter study — quantizing `‖W₀‖` would study a different method than the
one under test, so P2 rejects it outright rather than using it to reach larger models.

### What changed the priority order

In the first draft P3 was ranked first on the argument that it was the only project genuinely needing
scale. That argument now cuts the other way: the A6000 envelope brought real code RLVR in-house at
4B and 9B, so P3's *core* claim is testable without the earned pool, and the H100 ask is only for
the long-horizon extension. Meanwhile P1's ImageNet tier came back in-house too, at 400 hours, which
restored CVPR main track without spending anything.

The remaining escalations are therefore genuinely marginal — extensions to results that will already
exist. That is the correct state for an earned budget to be in, and it means **no project is blocked
on getting it**.


---

## Program review (second pass): risks, gaps, and what a reviewer will attack

This replaces the first-round review. It was written against the re-scoped sections 01, 015, 02–065,
07 and 075, without reference to the previous review's text. Section 065 (P6) has never been
reviewed and gets the most space.

The headline: **the program is in much better shape than it was.** The two problems that dominated
the first round — a compute plan that was wrong by an order of magnitude, and a flagship RL study
that could not reject its own null — are genuinely fixed, and the effective-learning-rate protocol
in P1 and P2 is now better than what most published papers in these areas do. What remains is a
smaller set of sharper problems, three of which will change what you do next week.

---

### Read this first

Ordered by how much each one changes next week's work.

1. **P6's Gate G0(c) — the gate that decides whether to buy the robustness half — is mis-specified,
   and P6's LR control never touches the arms where the robustness claim lives.** T0.4 sweeps LR on
   *clean, 60-epoch, non-adversarially-trained* models; T2.1/T2.2 (Fast AT, 50 epochs) get no sweep
   at all. Meanwhile G0(c) asks for a normalized-vs-standard gap of "~1.6–2.5 points robust
   accuracy" measured on T0.3's non-AT models, where robust accuracy at ε=8/255 is 0.0% for both
   arms — and any non-zero value is precisely the gradient-masking signature G0(d) exists to catch.
   The two halves of Gate G0 contradict each other. Fix before week 1: move a 2-arm × 3-seed Fast-AT
   pair (4.2 h, currently T2.1) into Tier 0, sweep LR on *that*, and gate on it. §1.3, §6.

2. **P3's 1.4-point primary MDE rests on a common-random-number correlation ρ = 0.7 that cannot be
   measured from the budgeted calibration design and is not plausible for RL fine-tuning.** T1.1/T1.2
   run one seed per (arm, LR) cell; you cannot estimate a cross-arm correlation from one seed. At the
   pre-registered conservative ρ = 0.4 the primary MDE is 1.9 points against a modal effect of
   +1…+2.5 — i.e. the underpowered-estimand problem has moved out of the interaction and into the
   simple effect. Add an explicit ρ̂ block (3 seeds × 2 arms at 40% length = 15 GPU-h) *before*
   committing T1.3/T1.4's 187.5 h. §3.2.

3. **Two cost models are optimistic in the same direction, and the program total is ~1,900–1,980
   hours, not ~1,800 — across two different cards.** P3 prices Qwen3.5-9B at 1.14× the 4B run
   (7.15 h vs 6.25 h) for a model with 2.25× the parameters; P2 prices the identical jump at 2.4×
   (5.8 h vs 2.4 h). One of the two sections is wrong and it is P3: Tier 2 is ~330 h, not 200, and
   P3 lands near 790 h. Separately, P1's FFCV throughput figures are ~1.5× optimistic and contradict
   P1's own A100→A6000 conversion factor; Gate 2→3a's 3,500 img/s threshold may sit above the
   A6000's *compute* ceiling, so an I/O gate will fire for a compute reason. §4.

4. **P6's central framing — "κ̂ is a better-specified NC1" — is not novel, and the section knows it
   but under-cites.** Per-class vMF fitting of classifier features with an estimated/learned κ is
   SIREN (NeurIPS 2022, uncited), KappaFace (cited), CIDER (cited), and the vMF-mixture-softmax
   line (Hasnat 2017 / NormFace); directional NC1 is HUG/GNC (arXiv:2303.06484, uncited here though
   P4 cites it); NC-under-adversarial-perturbation already has a paper. What is genuinely unclaimed
   is the **null-calibrated admissibility boundary and the `p`-sweep that validates it** — which is
   a methods paper, not a CVPR paper. Re-target P6 unless the robustness half survives Tier 0.
   §1.1.

5. **Three sections gate a go/no-go decision on a rank correlation computed on 4, 9, or 10 points.**
   P3 Gate 0B(c) requires Spearman ρ ≥ 0.8 over **four** `G` levels (that is "at most one inversion",
   p ≈ 0.04 at best); P1 Gate 1(ii) requires ρ ≥ 0.5 across "18 cells" that are really **9** once you
   pair the arms to form a gap (p ≈ 0.17); P6 G0(b) requires ρ ≥ 0.5 **and** permutation p < 0.05
   across **10** CIFAR-10 classes, which is internally impossible — at n = 10, p < 0.05 needs
   |ρ| ≥ 0.648. Re-specify all three, or move them to descriptive status. §3.5.

6. **P1's per-class feasibility table has a math error that propagates into the paper's headline
   methodological claim.** §1.2 says "R̄ has an O(√(p/n)) floor". It does not: for n uniform vectors
   on S^{p−1}, E‖x̄‖² = 1/n exactly, so the R̄ floor is n^{−1/2}, independent of p — which P4 §4.2
   and P6 §6.2 both state correctly. The p-dependence enters only after the Banerjee inversion
   (κ̂_null ≈ p/√n). P1 also uses n = 2,500 per class for CIFAR-10 where the correct figure is 5,000,
   which flips that row from "marginal" to "estimable" and puts it in direct conflict with P6's
   table for the same cell. §5.

7. **The null-calibrated measurement contribution is claimed as a headline by three sections
   simultaneously** — P1 §1.2, P4 §4.2/§4.4(f), P6 §6.2 — and P4 §4.4(f) is literally P6 §6.2's
   feasibility study with `n` swept instead of `p`. Decide the owner this week. Whoever ships first
   owns it; the other two cite it. §7.

8. **P5 is the one section asking for too many seeds.** 10 seeds in Tier 1 (§5.2, §5.3, §5.6 = 970
   runs), with slack "earmarked first for topping [Tier-2] blocks back up to 10 seeds". Keep 10 only
   for §5.2's reproduction, where matching Davidson's protocol is the point; 5 everywhere else frees
   ~17 GPU-h for the Tier-2 blocks that actually carry the science. §3.6.

---

## 1. P6 — novelty and validity audit

### 1.1 Is "κ̂ as a better-specified NC1" novel?

**No, not as stated.** §6.1's prior-art paragraph is honest as far as it goes and correctly names
CIDER, KappaFace, NECO and VCI. It misses the closest hits, and the ones it misses are the ones a
reviewer finds first:

- **SIREN (Du et al., NeurIPS 2022, "Shaping Representations for Detecting Out-of-Distribution
  Objects")** models class-conditional penultimate features as a **mixture of von Mises–Fisher
  distributions with per-class concentration**, and its test-time score is a **parametric vMF
  likelihood** under those fitted distributions. That is §6.1's estimand and §6.3's per-input score,
  published four years ago. It is not cited anywhere in 065. This is the single most damaging
  omission in the section.
- **HUG / Generalized Neural Collapse (arXiv:2303.06484)** already replaces the trace-ratio NC1 with
  a *directional* intra-class-variability measure on the hypersphere and argues explicitly that this
  is the right generalization when d < C−1. P4 §4.1 cites it; P6 does not, even though P6 is the
  section making the NC1 claim.
- **vMF-mixture interpretations of normalized softmax** (Hasnat et al. 2017; NormFace, which P1 §1.1
  cites) already establish that a cosine classifier *is* a shared-κ mixture-of-vMF likelihood. So
  "on normalized features the principled directional form of within-class variability is the vMF
  concentration" is a restatement, not a discovery.
- **Neural collapse under adversarial perturbation and in adversarially trained networks** is
  already a paper (the "robustness of neural collapse / neural collapse of robustness" line): NC
  metrics decrease on both clean and perturbed points under AT, and NC1 collapse is weaker on
  perturbed points than on clean ones. That is the qualitative result §6.6 would report. Cite it and
  state the delta, which is the null calibration, not the phenomenon.

**What is actually unclaimed, and it is worth having:**

1. The **admissibility boundary** — the operating rule `R̄ > 0.19` / `0.05 < R̄ ≤ 0.19` / `R̄ ≤ 0.05`
   with a matched null in every cell, and the demonstration that per-class NC1-as-κ is *unmeasurable*
   on exactly the benchmarks where the NC literature reports it most confidently (CIFAR-100,
   Tiny-ImageNet, wide features, ~500 images/class).
2. The **`p`-sweep at fixed backbone and fixed n** (T1.4). This is the direct test of "κ/p governs
   accuracy, not p" and it is the single most defensible new measurement in the whole program. It
   costs 7 GPU-h and it should be the section's Figure 1.
3. The **superclass-pooling + narrow-projection-head fix**, which turns the negative result into a
   constructive one.

**Verdict.** Items 1–3 are a real contribution. They are a **measurement/methods paper — TMLR or an
ICLR methods slot — not CVPR.** §6.1's own self-assessment ("the metric framing alone is thin") is
correct and should be believed. The CVPR framing depends entirely on the robustness half, and the
robustness half is the weak half. Recommend: pre-register the venue decision as a function of the
Tier-0 outcome, rather than assuming CVPR in 075 and in the overview.

**One thing the section should add and does not, because it would be the paper's teeth.** §6.2
demonstrates that per-class κ is unmeasurable at CIFAR-100 scale. It never *audits an existing
claim*. P4 §4.4(e) does exactly this for the isotropy literature. The analogue here is nearly free:
take 10–20 released CIFAR-100 / Tiny-ImageNet checkpoints (timm, RobustBench), recompute the
published NC1 numbers, put a matched null next to each, and produce a table of which published NC1
claims survive. That is ~1 GPU-h and it converts "here is a boundary" into "here are the published
numbers that fall on the wrong side of it." Without it, §6.2 is a rule nobody has to obey.

### 1.2 Is the adversarial-detection half credible against modern baselines?

**No, as currently specified.** Three problems, in descending order.

**(a) The baselines are OOD-detection baselines, not adversarial-detection baselines.** MSP (2016),
ODIN (2017), Mahalanobis (2018) and energy (2020) are the standard *out-of-distribution* suite.
Adversarial-example detection is a different and much more hostile literature, and the reference
points a reviewer will name are **Carlini & Wagner 2017 ("Adversarial examples are not easily
detected: bypassing ten detection methods")** and **Tramèr, ICML 2022 ("Detecting adversarial
examples is (nearly) as hard as classifying them")**. Tramèr's reduction is the fatal one: a robust
detector at distance ε implies a robust classifier at ε/2, and he revisits 13 detector defenses and
shows that in 11 of them the claimed detection results would imply classifiers far beyond SOTA.
Neither paper appears in §6.3. Any reviewer who works in this area will open with them, and "we beat
Mahalanobis on AUROC" is not a response.

To be fair, §6.3 does the two things that matter most: it runs an **adaptive attack** (100-step PGD
on `L_CE − λ log p_vMF`, 5 λ values) and it states plainly that the per-input score is cosine-to-
class-mean with a calibration wrapper. That honesty is worth a lot. But the framing has to change:
this is not "does κ detect adversarial inputs", it is "at matched cost, and in the `n_class < p`
regime where Mahalanobis' class-conditional covariance is singular, a directional score is
well-posed where the covariance score is not." That is an **estimation-well-posedness** result, and
§6.3 already says so in its efficiency paragraph. Lead with it, cite Tramèr up front as the reason
the paper does *not* claim a security result, and the section becomes defensible.

**(b) There is no distribution-shift control.** If the vMF likelihood fires on adversarial inputs, a
reviewer will immediately ask whether it fires on **CIFAR-10-C** at matched accuracy degradation. A
detector that cannot distinguish an L∞ adversarial perturbation from Gaussian noise or JPEG
compression is a shift detector, not an adversarial detector. CIFAR-10-C is a download and the score
is inference-only: this costs under 0.5 GPU-h and its absence is the most predictable reviewer
demand in the section.

**(c) There is no held-out attack.** All four attacks in T1.5 are in the design set. A detector
tuned — even implicitly, via layer-aggregation choices — on FGSM/PGD/C&W/AutoAttack must be shown to
transfer to an attack it never saw. Free, since the attacks are already implemented.

**Power note on §6.3.** G1(e)'s threshold (vMF AUROC ≥ Mahalanobis − 0.02 on ≥3/4 attacks) is the
one detection criterion that is adequately powered on the *evaluation* axis — with 1,000 images the
paired AUROC-difference SE is ~0.005, so 0.02 is ~4 SE. It is **not** powered on the training-seed
axis: T1.5 runs on "4 models" with no seed replication, so a 0.02 AUROC difference attributable to
which seed trained the encoder is invisible. Say so, or share encoders across the comparison so the
contrast is within-encoder.

### 1.3 Are P6's gates real?

Mostly yes, and in two places they are the most rigorous gates in the program — the Tier-0 masking
pre-screen with numeric thresholds fixed in advance, the black-box Square-500 diagnostic in the
*screen* rather than only in the Tier-2 checklist, the day-3 tripwire on (d1)/(d4), the standing
publication rule on sub-0.25-L2 radii, and the cheapest-first ordering of G2a → G2b → G2c. That is
genuinely good design and it should not be watered down. But four specific defects:

**G0(a) is vacuous.** It requires penultimate-layer `R̄` on RN-18/CIFAR-10 train to cross **0.19**
before end of training. At TPT with L2-normalized features, within-class mean pairwise cosine on a
converged ResNet-18/CIFAR-10 is routinely 0.7–0.9, giving R̄ ≈ 0.84–0.95. This passes with
near-certainty. The second clause ("exceeds the matched null by ≥ 10 null SDs") is subsumed by the
first by a factor of ~40: sd(R̄_null) ≈ 1/√(2np) = 4.4e−4 at n = 5000, p = 512, so R̄ = 0.19 is
already ~400 null SDs above the null mean of 0.014. The gate also does not say whether R̄ is
per-class or pooled — and pooled R̄ measures the common-direction/cone effect (NC2-adjacent), not
NC1. Specify per-class, and set a threshold that can fail.

**G0(b) is arithmetically impossible as written and is computed on 10 points.** Spearman across the
**10 CIFAR-10 classes**. The gate demands |ρ| ≥ 0.5 *and* permutation p < 0.05; at n = 10 the 5%
two-sided critical value is |ρ| ≈ 0.648, so the stated ρ threshold is inoperative and the real
threshold is 0.648 — which the section never says. Worse, this is one of four *conjunctive*
conditions that can kill the project at 9.9 h. Fix: run the difficulty correlation on **CIFAR-100's
100 classes** (T1.1), where n = 100 gives real power, and demote the CIFAR-10 version to descriptive.

**G0(c) gates the robustness half on a quantity that is either zero or an artifact.** It requires the
normalized-vs-standard difference in T0.3 to survive T0.4's confirmation at "~0.9 points clean,
~1.6–2.5 points robust". T0.3 trains **standard, non-adversarial** ResNet-18s for 200 epochs. Robust
accuracy of such models under PGD at ε = 8/255 is 0.0% for both arms. So either the gate collapses
to a clean-accuracy comparison — which is not evidence about robustness at all — or the normalized
arm shows non-zero robust accuracy without AT, which is exactly the gradient-masking signature that
G0(d) is built to detect. **G0(c) and G0(d) are in direct tension.** The fix is the one in the
"Read this first" list: hoist a small Fast-AT block into Tier 0 and gate on that.

**The conjunction structure gives a high false-stop rate.** G0 requires (a) AND (b) AND (c) AND (d),
and (d) is itself a five-way conjunction of noisy diagnostics evaluated in ≥2/3 seeds. If each of the
five masking diagnostics has even a 10% chance of a spurious fail, P(all five pass) ≈ 0.59; fold in
(b) at n = 10 and the project has a substantial chance of being killed at 9.9 h by noise rather than
by evidence. Gates that fail cheaply are the right design, but a gate that fails *wrongly* and
cheaply is worse than no gate. Recommend: make (d) the blocking condition — it is the one with a real
decision attached — and make (a)/(b) reportable diagnostics that inform scope rather than kill it.

### 1.4 §6.5 — the input-space route inherits the composition problem it was chosen to avoid

The rewrite of §6.5 is the best piece of self-criticism in the whole plan: G2a is priced at 0.2 h,
runs in week 1, is *expected to fail*, and the standing publication rule pre-commits to never
printing a 1e−3 radius as a certificate. Keep all of that.

But the inversion — "input-space vMF smoothing over normalized image directions is the primary
route ... it yields a genuine, uncomposed input-space certified set ... and it needs no Lipschitz
assumption at all" — is not right. Smoothing over **normalized image directions** certifies a
geodesic ball on S^{3071}, in a space where the image's norm has been discarded. The threat model
everyone else uses is an L∞ or L2 ball in *pixel* space. Mapping a geodesic radius θ on the
normalized-image sphere back to a pixel-space L2 radius requires bounding ‖x‖ and handling the
normalization map's own behaviour — i.e. a composition with the same structure as the latent-space
composition the section rejects, just with a friendlier constant (an angular radius θ maps to roughly
2 sin(θ/2)·‖x‖, which for CIFAR may actually clear 0.25 — but it has to be computed, not assumed).
Two required changes: (i) state the normalized-sphere → pixel-space composition explicitly and
compute the constant; (ii) apply the standing 0.25-L2 rule to this route too, not only to the latent
route.

Two smaller notes. The claim that N = 1e4 "caps the maximum certifiable radius at roughly 60% of the
N = 1e5 ceiling" is conservative: the Clopper–Pearson lower bound at N = 1e4 gives p_A ≈ 0.99931
→ Φ⁻¹ = 3.20, versus 0.999931 → Φ⁻¹ = 3.81 at N = 1e5, i.e. **84%**, not 60%. That errs in the safe
direction, so it is fine, but get the number right in the paper. And G2c's paired McNemar design
(MDE ~6 points certified accuracy, ~0.02 ACR, pre-registered, with the fallback paragraph written
*before* the run) is the single best-specified decision rule in the entire six-section plan. Copy
that pattern into the other sections.

### 1.5 P6's seed and power holes

The brief asks whether P6 states a power analysis. It states two — G0(c)'s MDE (0.9 clean /
1.6–2.5 robust at 3 seeds, from a correctly-derived Cohen's d ≈ 3.1) and G2c's paired MDE. Both are
honest. But **the primary Tier-2 robustness claim has no power analysis at all**, and two Tier-1
items are single-seed while the gates they feed demand seed CIs:

- **T1.7 has one seed per cell.** "Three displacement levels × two noise families × train-time and
  test-time is 6 training runs" — that is 3 × 2 = 6 cells, one run each. Yet **G1(f)** requires "a
  vMF-vs-Gaussian robust-accuracy gap ≥ 2 points at some displacement level, **outside the seed
  CI**". There is no seed CI. This also contradicts §6.7's "3 seeds on every CIFAR-10/CIFAR-100 arm".
- **T1.2 (Tiny-ImageNet) and T1.3 (ViT-S) are 1 seed**, and §6.8 claims as an offset that it
  "dropp[ed] the two single-seed Tiny-ImageNet arms, which carry no seed CI and therefore could
  never have supported a robustness claim" — while the Tier-1 table still lists them and §6.7 still
  says "1–2 on Tiny-ImageNet and ViT-S". One of those three statements is stale.
- **T1.4 (the `p`-sweep), the section's best experiment, is 2 seeds** across 5 dimensions, and G1(g)
  plus the Tier-2 unlock both depend on it. Two seeds on the *most* defensible item and three on the
  least defensible one is backwards. Move a seed from T1.1 to T1.4.
- **T2.1/T2.2/T2.4 have no MDE.** The headline — normalized vs standard AutoAttack robust accuracy at
  matched clean accuracy — is 3 seeds per arm, and AA robust-accuracy seed SD at ResNet-18/CIFAR-10
  Fast-AT scale is typically 0.8–1.5 points, giving MDE ≈ 2.5–4.7 points. State it. Against the
  literature's typical claimed normalization effect (1–3 points), the tier is probably underpowered;
  if it is, say so in the caption rather than discovering it at review.

### 1.6 P6 compute inconsistencies

- **CIFAR-100 is priced at 2.14× CIFAR-10 for the same architecture and the same 50,000 images.**
  T0.3 (CIFAR-10, 200 ep, 2 arms × 3 seeds) = 4.2 h; T1.1 (CIFAR-100, 200 ep, 2 arms × 3 seeds) =
  9.0 h. T2.1 (Fast AT, CIFAR-10) = 4.2 h; T2.2 (Fast AT, CIFAR-100) = 9.0 h. Same images, same
  ResNet-18, only the head width differs. Either CIFAR-10 is 2× underestimated or CIFAR-100 is 2×
  over-budgeted; ~9 h hangs on it.
- **The AT cost model is internally inconsistent by ~3×.** §6.6 says PGD AT "lands at roughly 0.7
  GPU-h per ResNet-18/CIFAR-10 run against **~7 h** for the PGD version"; the table prices T2.3
  (PGD-7 AT, 2 runs) at **5.0 h**, i.e. 2.5 h/run. And T0.3's clean 200-epoch run at 0.7 h/run
  implies Fast AT at 50 epochs should be ~0.35 h, not 0.7. State epochs and the per-epoch multiplier
  on every AT line.
- **§6.4's matched-displacement κ values mix two approximations.** κ = 25,550 and 2,555 at A = 0.99
  and 0.9 come from the large-κ form (p−1)/(2(1−A)); κ = 341 at A = 0.5 comes from Banerjee. Banerjee
  at A = 0.9 gives 2,421 — a 5% disagreement. Since the entire experiment is "at matched A_p", solve
  A_p(κ) = A numerically through the CUSF path. You have it, it costs nothing, and mismatching the
  displacement by 5% is exactly what the vMF-vs-Gaussian comparison is sensitive to.

---

## 2. What is now FIXED — the parts that are safe

Say this out loud, because three of these were the previous round's most damaging findings.

- **P3's estimand problem is genuinely fixed, and fixed the right way.** §3.6 does not buy seeds; it
  fixes the analysis and then **demotes the interaction to a secondary, explicitly-underpowered,
  non-gating quantity** with its MDE printed beside it, and pre-commits the write-up for a null. The
  paragraph beginning "So the interaction is demoted" is the best piece of statistical writing in the
  plan. The 12-seed over-correction is gone (5 at 4B, 3 at 9B), and §3.8's cut order — ranks and
  steps before seeds, seeds last — is right. This is now a defensible design; its remaining problem
  is ρ (§3.2), not the estimand.
- **The compute plan is no longer incoherent by 3–16×.** The per-section budgets now sum to within
  ~10% of the escalation figure instead of 3×. The overview's "Read this first" item 4 (~20,900 GPU-h
  vs ~6,250) is stale and should be deleted.
- **P2's effective-LR control is now best-in-class.** 47.5 h of per-arm sweeps out of 571 (8.3%), the
  selected-LR-vs-rank curve as a *diagnostic figure* rather than an appendix table, rsLoRA as a
  scaling control, a matched-effective-LR LoRA arm, and T2.3(a) hard-renormalization as an explicit
  mechanism-killer. Nobody in the PEFT literature does this. Lead with it.
- **P1's effective-LR control is nearly as good** and covers more axes: per-arm (η, λ) sweeps in
  *both phases* at *every* tier, phase-2 selection on new-class accuracy only and never peeking at
  old-class accuracy (an excellent and rarely-observed detail), Kosson angular-update matching, a
  norm-matched control, normalization-without-tangent-projection, a Riemannian control, and
  wall-clock reported alongside epochs. §1.1(a)–(f) is the reference implementation for the program.
- **The κ constraints are respected almost everywhere.** The n = 1 / R̄ = 1 division-by-zero trap is
  stated and honoured in P1 §1.2(6), P2 §2.7(f), P3 §3.7, P4 §4.1, P5 §5.7 and P6 §6.3 — six for six.
  The CUSF/torch log-Bessel requirement at p ≥ 1024 is stated in all six, with correct Bessel orders.
  The matched null is mandatory in all six. P2 §2.7(f) additionally gets the *right* null for a
  trajectory statistic — an isotropic random walk with step lengths resampled from the observed
  distribution, not iid uniform — which is a subtlety nobody had to catch.
- **The stability–plasticity frontier as P1's primary object** is correct, costs nothing, and Risk 1b
  ("the arms share a frontier") is exactly the right null to name in advance.
- **P4's §4.4(c) rewrite is correct.** The known-mean estimator has no leading-order upward bias and
  κ̂_null ≈ √(p/n); the arithmetic checks out (at p = 512, κ ≈ 60, n = 10⁷: A_p ≈ 0.116,
  A′_p ≈ 1.9e−3, sd(κ̂) = (nA′)^{−1/2} ≈ 0.007, relative precision ~1e−4 as claimed). Recognising
  that this makes an NHST meaningless and pre-registering a **tolerance band** instead is exactly
  right and will survive review.
- **P6's gate ordering, standing publication rule, and Tier-0 black-box screen** (§1.3) are real
  improvements over the version they replaced, and the G2c paired-McNemar design should be copied.

---

## 3. Statistical power and seed-count realism — both directions

### 3.1 Seed counts are, with one exception, now in line with practice

| Section | Seeds | Verdict |
|---|---|---|
| P1 MLP tier | 5 partitions × 5 seeds | Acceptable — partitions are replication, not seeds, and runs cost 0.035 h. 5 × 3 would lose almost nothing |
| P1 ResNet/CIFAR | 3 orders × 3 seeds | Right |
| P1 ImageNet | 2 orders × 3 seeds | Right, and honestly labelled "confirmation, not establishment" |
| P1 PTM | 5 / 3 | Right |
| P2 vision | 3 | Right, and the VTAB-19 replication is the pattern the rest of the program should copy |
| P2 language 4B / 9B / GSM8K | 5 / 2 / 3 | Right, and the 2-vs-5 argument at 9B is correct |
| P3 vision / 4B / 9B | 5 / 5 / 3 | Right |
| P4 nGPT ladder | 3 / 2 / 1 | Right for a ladder whose object is a trend |
| **P5 Tier 1** | **10** | **Too many.** §3.6 |
| P6 CIFAR arms | 3 | Right — except T1.7 (1), T1.2/T1.3 (1), T1.4 (2). §1.5 |

Nothing anywhere asks for more than 5 except P5, and the 12-seed design is gone. That part of the
over-correction is genuinely closed.

### 3.2 P3 — the CRN assumption is the whole ball game and it is not verifiable as budgeted

§3.6's four pooling mechanisms are all legitimate, and the section is unusually honest about which of
them *don't* buy power ("VTAB-19's 19 tasks are 19 independent training runs; MBPP-Plus's 378
problems are 378 evaluations of a single training run" — correct, and exactly the mistake a lazier
plan would have made). But mechanism 2, common random numbers, is doing most of the work: the primary
MDE moves from 2.9 (naive) to 1.4 almost entirely on ρ = 0.7.

Three problems.

1. **ρ = 0.7 is a stochastic-simulation number, not a deep-learning number.** CRN works when the
   random stream drives the same events in both systems. LoRA and DeLoRA-PC have different parameter
   shapes and different early dynamics *by construction* (P2 §2.1 point 2: DeLoRA-PC has a non-zero
   gradient at init where LoRA's A is frozen for the first step). Sharing data order helps; sharing
   "rollout RNG streams" is nearly meaningless once the policies' logits diverge, because inverse-CDF
   sampling from different distributions yields different tokens within a handful of steps. A
   realistic planning value is ρ ≈ 0.2–0.4.
2. **ρ̂ cannot be measured from the budgeted design.** §3.6 says "ρ̂ is measured from the calibration
   runs before unblinding" and §3.9 says "if ρ̂ < 0.55 the published MDEs are revised upward before
   unblinding". T1.1 is 4 arms × 4 LRs × **1 seed**; T1.2 likewise. A cross-arm correlation across
   seeds requires ≥3 seeds per arm. As budgeted, ρ̂ first becomes estimable partway through T1.3 —
   after ~40% of a 125-hour block.
3. **At ρ = 0.4 the primary is underpowered against the modal effect.** MDE 1.9 (the section's own
   number) against Project 2's Case-B modal SFT gap of +1…+2.5, with the RLVR gap presumed comparable
   or larger. That is the same shape of problem the first review found, relocated.

**Fix, and it is cheap.** Insert a ρ-estimation block: 2 core arms × 3 seeds at 40% run length = 6
runs × 2.5 h = **15 GPU-h**, before T1.3. Publish ρ̂ and the revised MDE. If ρ̂ < 0.5, the
pre-registered response should be written *now* — the honest options are (a) accept a 1.9-point MDE
and rewrite Gate 1(a)'s threshold accordingly, or (b) add 2 seeds at 4B (+62.5 h with twins and
evals), which keeps you at 7 seeds and is already at the edge of practice. Do not discover this at
step 300.

**A second, structural point.** The well-powered primary pools r ∈ {1, 2}. At r = 1, per-component
scaling **is** a global scalar — P2's own `test_reduces_to_delora_at_r1` proves it. So half of the
primary contrast's replication is a measurement of **DeLoRA vs LoRA**, i.e. of Bini et al.'s
published method, not of the novel delta. The novel delta (H3, r = 2 only, MDE 1.6) is the thing
Gate 1(b) blocks on, and it is the *less* powered of the two. State that asymmetry explicitly: "we
power a claim about prior art to 1.4 points and gate on a claim about our own delta at 1.6."

### 3.3 P2 — the MDE arithmetic checks out, but the estimand should be named

`sqrt(τ²/K + σ²/(KS)) × 2.80`: vision, τ = 0.8, σ = 0.7, K = 19, S = 3 → 0.576 ✓ (stated 0.58).
Language 4B, τ = 1.0, σ = 1.2, K = 8, S = 5 → 1.123 ✓ (stated 1.12). 9B, S = 2 → 1.30 ✓. The 9B
S = 5 row is stated as 1.06 but the same model gives 1.12 — a 5% table inconsistency that, notably,
*strengthens* the section's argument: 2 → 5 seeds buys 0.18 points, not 0.24.

The substantive point: including `τ²/K` makes the estimand **"the effect on a randomly drawn task
from this population"**. For the claim actually being made — "per-component beats global on the
commonsense suite" — the fixed-task estimand drops the τ² term and gives MDE = 2.80·σ/√(KS) = 0.53 at
4B and **0.84 at 9B with 2 seeds**, materially better than 1.30. Which estimand you want is a real
choice (random-task is more conservative and more generalizable), but the section must name it,
because a reviewer who does the arithmetic the other way will think the test is mis-specified. This
also sharpens §2.8's central argument: under the fixed-task estimand, seeds *do* keep buying power at
9B, and "τ²/K dominates and no seed count removes it" is estimand-dependent, not a fact.

The commitment to re-issue the MDE table from the measured `r=1` null spread before Gate 1 (§2.12) is
the right practice — but note that a spread estimated from **3 seeds** has ~50% coefficient of
variation, so compute it as the sd of the paired per-task difference across 19 tasks × 3 seeds (57
paired observations), not as the sd of 3 task-averaged numbers. Same for Gate 0's "2× the `r=1` null
spread".

### 3.4 P1 — a power *note*, but no MDE anywhere, while every gate is stated in accuracy points

§1.1 says: "at n = 6 cells (ImageNet), a paired Wilcoxon has ~0.8 power only for effects ≳ 1.5
σ_paired". It never states σ_paired. Every gate in the section is a point threshold — Gate 0 at
≥3.0 points, Gate 1 at ≥2.0, Gate 2→3b at ≥2.0, Gate 2→4 at ≥2.0 — and none carries a variance
estimate, so **none can be checked for reachability in advance.** Back of envelope: at 25 paired
cells with a CIL paired-difference sd of ~2 points, Gate 0's MDE is ~1.1 points and the 3.0-point
threshold is comfortably powered; at the 6-cell ImageNet tier with σ_Δ ≈ 1.5, the MDE is ~1.7 points
and the 2.0-point threshold is marginal. Both are acceptable answers — but P1 should print the table,
because "we set our gate at 2.0 points" invites "what could you have detected?", and Tier 3 is 53.5%
of the section's budget.

**Concrete ask:** add an MDE column to §1.7.6, populated from Tier 0's measured σ_Δ before Tier 2
starts. Tier 0 produces 25 paired cells; you will know σ_Δ in week 1.

### 3.5 The tiny-n rank correlations

| Gate | Statistic | n | Stated threshold | Actual 5% critical value | Verdict |
|---|---|---|---|---|---|
| P3 Gate 0B(c) | Spearman over `G ∈ {2,4,8,16}` | **4** | ρ ≥ 0.8 | ρ = 1.0 (p = 0.042) | "at most one inversion"; a coin flip |
| P1 Gate 1(ii) | Spearman, R̄ gap vs accuracy gap over "18 cells" | **9** (arms pair away) | ρ ≥ 0.5 | ρ ≈ 0.683 | p ≈ 0.17 at threshold; no power |
| P6 G0(b) | Spearman, κ̂_c vs per-class error | **10** | ρ ≥ 0.5 **and** p < 0.05 | ρ ≈ 0.648 | internally inconsistent |

Two of the three (P1, P3) are also *mechanism* claims — co-variation and dose–response — which is
precisely where a null matters most and where an underpowered test is most dangerous, because it
converts "we could not tell" into "the mechanism is absent". Fixes: **P3**, extend `G` to 6 levels
(marginal cost ~2 GPU-h at 0.065 h/run) or replace the rank test with a regression slope on the
4 levels × 5 seeds = 20 observations with a CI; **P1**, run the co-variation regression on 9 paired
cells × 5 seeds = 45 observations with cell as a random effect, not a 9-point Spearman; **P6**, move
it to CIFAR-100 (n = 100 classes).

### 3.6 P5 — the one section over-buying seeds

§5.7's Tier 1 is 1,054 runs: §5.2 reproduction (100 runs, 10 seeds), §5.3 dimension wall (150 runs,
10 seeds), §5.6 S-VGAE grid (**720 runs**, 10 seeds). §5.7 then earmarks the 25-hour slack "first for
topping [Tier-2] blocks back up to 10 seeds". This is precisely the over-correction the program was
told to stop making.

- **Keep 10 seeds in §5.2 only.** Davidson et al. report 10 runs per cell; matching their protocol is
  the point of a reproduction and is a legitimate reason.
- **Cut §5.3 and §5.6 to 5 seeds.** The published S-VAE/N-VAE gap at d = 40 is ~1.9 nats against a
  reported sd of 0.30–0.34; at 5 seeds the SE of the paired difference is ~0.2 nats, i.e. ~9σ. Ten
  seeds buys nothing. Savings ≈ 7.5 + 5.5 ≈ **13 GPU-h**, plus ~2 more if §5.3's extreme grid drops
  from 5 seeds to 3.
- **Spend it on Tier 2**, which is where §5.5's falsifying ablations live and which is currently at 3
  seeds — the exact inversion of where the seeds should be.

P5 also states **no MDE anywhere**. G1's criterion ("≥ 2 s.d. separation at d ≥ 256, across 10
seeds") is at least variance-referenced, which beats a bare point threshold, but the paper will need
the gap in nats with a CI, so compute it now.

---

## 4. Compute honesty

### 4.1 The totals do not sum to 1,800, and they mix two cards

| Project | Section's own total | 075 / overview | Card |
|---|---|---|---|
| P1 | 400.0 (= 10 + 30 + 57 + 214 + 89 ✓) | 400 ✓ | A6000 |
| P2 | 571.1 committed / **656.8 budgeted** | **475** ✗ | A6000 |
| P3 | 656.3 (= 5.2 + 42.9 + 407.8 + 200.4 ✓) | 662 (stale by 6 h) | A6000 |
| P4 | 60.1 + 12 = **72.1** | **66** ✗ | 3090 |
| P5 | 72 + 35% = 97 ✓ | 97 ✓ | 3090 |
| P6 | 82.4 + 20% = 98.9 ✓ | 98 ✓ | 3090 |
| **Total** | **1,895 committed / 1,981 budgeted** | **~1,800** | **mixed** |

Two problems. First, 075 says "All figures are single-A6000 hours unless noted" and then lists P4, P5
and P6, all of which are costed in **3090-hours**. P1 §1.7.1's own conversion is 1 A6000-h ≈ 1.22
3090-h, so the three 3090 sections are being counted at ~82% of their A6000-equivalent weight. Convert
properly and the program is ~1,850 A6000-equivalent hours committed, ~1,930 budgeted.

Second, the "**~1,800 GPU-hours on one card**" framing hides the calendar. At ~1,900 hours and 70%
realistic utilization, one card is **~113 days of pure GPU time**, before any of the CPU work — and
the six sections' calendars (P1 18 weeks, P2 18 weeks, P3 ~28 days at full occupancy, P4 12 weeks,
P5 7–8 weeks, P6 3 weeks) are each written as though that section owns the card. Serialized on one
A6000 at 0.6–0.8 research FTE, this is a 9–14 month program, not the overlapping schedule the sections
imply. Say so once, in the overview, and pick an order.

**"Every gate opens for ≈42 GPU-hours" survives** and is the right number to act on — but its
per-project entries are stale: 075 says P3 Tier 0 = 6.8 (section says 5.2) and P6 Tier 0 = 8.0
(section says 9.9). The total is ~42 either way, by coincidence.

### 4.2 P1's FFCV numbers — checked specifically, as asked

The epoch-equivalence argument in §1.7.4 is **correct and is the best idea in the section**:
10 tasks × 16 epochs × 128k images = 20.48M presentations = 16 full-ImageNet-epoch-equivalents, so a
whole 10×100 CIL sequence costs what one 16-epoch joint run costs. Keep that argument in the paper.

The throughput numbers underneath it do not hold up.

- **"FFCV reaches 75% top-1 with ResNet-50 in ~2.7 A100-h."** FFCV's headline is 8×A100 for ~20 min,
  which is where 2.7 comes from. But 2.7 A100-h at FFCV's actual per-GPU ResNet-50 throughput
  (~2,500–3,000 img/s at reduced resolution with progressive resizing) buys **~20 epochs**, and
  FFCV's own published rn50 ladder puts 16–20 epochs at ~71%, not 75%. The 75% config is ~40 epochs,
  i.e. **~5–6 A100-h ≈ 13–15 A6000-h**, not 6–8.
- **The img/s figures contradict P1's own conversion factor.** §1.7.1 fixes 1 A100-h ≈ 2.5 A6000-h,
  i.e. A6000 ≈ 0.4× A100. At 2,700 img/s on an A100, ResNet-50 on an A6000 is ~1,080 img/s, not the
  **1,600** stated in §1.7.4. By the section's own 3× rule, ResNet-18 is then ~3,200 img/s, not
  **4,800**. That is a uniform ~1.5× optimism, and it is internally checkable without buying anything.
- **Consequence.** Per-run costs move from 1.5 → ~2.2 h (R18) and 4.5 → ~6.6 h (R50). Tier 3 moves
  from 214 → **~305 GPU-h** and P1 from 400 → **~490 A6000-h (~600 3090-h)**. Still inside the
  1,000-h ceiling, but the "600 hours of headroom" narrative in §1.7.6 shrinks to ~500, and the 3090
  path loses most of its slack — which matters, because three of the four headroom priorities
  (32-epoch re-run, second ResNet-50 order, 5 seeds on the 10×100 primary) are the ones a reviewer is
  most likely to force.
- **Gate 2→3a's threshold is in the wrong units and may be unsatisfiable.** "Require ≥ 3,500 img/s
  and ≥ 85% GPU util ... Below 2,000 img/s you are I/O bound." If the A6000's *compute* ceiling for
  ResNet-18 at 176² is ~3,200 img/s, a machine with a perfect data pipeline fails the gate and the
  failure is attributed to storage. **Fix, 15 minutes:** in T3.0, first measure throughput on
  synthetic tensors with the dataloader bypassed to establish the compute ceiling C, then set the I/O
  gate at 0.85·C rather than at an absolute 3,500. This is week-2/3 work per the timeline, so it is
  genuinely next-week.

The 67%-at-16-epochs figure for ResNet-18 matches FFCV's published rn18_16_epochs config and is fine.
The honesty caveat about short-schedule ImageNet being a *relative* comparison, plus T3.5's 32-epoch
ranking-invariance control run early and before the ResNet-50 and B500 spends, is the right structure
and should be kept exactly as written.

### 4.3 P3's 9B pricing is not physical

§3.8 prices a 4B GRPO run at 6.25 h (37.5 s/step) and a 9B run at 7.15 h (43 s/step) — a **1.14×**
ratio for a **2.25×** model. Both phases of a GRPO step scale with N: decode is bandwidth-bound and
reads 2.25× the weight bytes per token, and the LoRA update is compute-bound at ~6ND. Working it
through with the section's own numbers: at 9B, generating 32,768 tokens at ~800 tok/s (the
bandwidth-scaled value; the section's 1,100–1,300 implicitly assumes only a 1.5× slowdown) is ~41 s,
and the update at 6 × 9e9 × 32,768 / 54e12 is ~33 s → **~70–75 s/step**, or ~11.7–12.5 h/run.

Tier 2 becomes **~330 GPU-h** instead of 200.4, and P3 lands near **790 GPU-h** — inside the ceiling,
but consuming the entire R6 contingency before R6 has happened. Note that **P2 gets this scaling
right** for the same two models on a nearly identical workload (5.8 h at 9B vs 2.4 h at 4B = 2.4×).
Two sections of the same program disagree by 2× on the cost of the same model. Reconcile in week 1 by
measuring one 9B step — a ten-minute job.

### 4.4 Anything quietly multi-thousand-hour?

Nothing inside the six sections. Two things outside them:

- **Section 07 still carries a "P3 — nViT / nCNN / nTransformer" project at 4,000–7,000 A100-hours**
  (≈ 10,000–17,500 A6000-h), and the overview's translation table describes it as "a **new** proposal
  ... worth keeping as a candidate". It has no section, no gate, and no budget line, and it is 5–9×
  the entire rest of the program. Either give it a Tier 0 and a gate, or delete it from the contents.
- **P1's D3** (full-fidelity 88-epoch ImageNet CIL) is honestly costed at ~1,180 A6000-h ≈ 340 H100-h
  and correctly deferred. Fine — but if T3.5's ranking flips *and* the corrected throughput of §4.2
  applies, the "+54 h to re-run at 32 epochs" fallback becomes ~+78 h and the headroom no longer
  covers it alongside the other three headroom priorities.

---

## 5. The κ measurements

The four hard constraints are respected in all six sections at the level of stated policy. The
violations are in the arithmetic, in one operating rule, and in one cross-section disagreement.

**Violations and errors:**

1. **P1 §1.2, "R̄ has an O(√(p/n)) floor"** — wrong. For n uniform unit vectors on S^{p−1},
   E‖x̄‖² = 1/n exactly, so the R̄ floor is **n^{−1/2}, independent of p**. P4 §4.2 states this
   correctly ("R̄ ≈ n^{−1/2} to leading order, essentially independent of p") and P6 §6.2 uses it
   correctly (`R̄_null ~ n^{−1/2}`; its numeric 0.014 at n = 5000 is exactly 1/√5000). The
   p-dependence appears only *after* the Banerjee inversion: κ̂_null ≈ p/√n. P1 has conflated the two,
   and the sentence sits in the row that justifies not computing per-class κ̂_c on CIFAR-100 — the
   conclusion is right, the stated reason is wrong, and a directional-statistics reviewer will notice
   in the first pass of a paper whose contribution *is* directional statistics.
2. **P1 §1.2, CIFAR-10 per-class n = 2,500** — CIFAR-10 has **5,000** training images per class, and
   P1's own ResNet-18 rows use 5,000 ("8 cls × 5,000 = 40,000"). Correcting it moves n/p from 4.9 to
   9.8 and flips the row from "Marginal" to estimable-at-the-10p-edge — which is exactly what P6's
   table says for the same cell. Internal to P1 *and* cross-section.
3. **P1 §1.2 vs P6 §6.2, a direct contradiction about the program's own headline measurement claim.**
   P1: "ImageNet-1k with ResNet-18 (1,300 train images/class at p = 512) is the **only** setting in
   the entire program where per-class κ̂_c is even arguably estimable." P6: CIFAR-10 train at p = 512,
   n = 5,000/class is "yes (at the 10p edge)"; RN-18 + 128-d head at n/p = 39 is "yes, comfortably";
   CIFAR-100 superclasses + 128-d head at n/p = 19.5 is "yes". P6 is right and P1's "only setting in
   the entire program" is false. Since both sections plan to publish the feasibility boundary as a
   contribution, they cannot disagree about which cells fall inside it.
4. **P6 §6.2's operating rule is stated on R̄ alone and therefore admits n < p cells.** "Report
   `kappa_hat` only where `Rbar > 0.19`" — at CIFAR-100 with n = 500 < p = 512, a converged
   classifier's per-class R̄ is ~0.85, so the rule says "report it" while the table two paragraphs
   later says "no". The rule must be a **joint** condition on R̄ *and* n/p (the table's verdicts are
   already computed that way). One sentence to fix — but it is the section's stated operating rule and
   it is what a reader will lift out of the paper.
5. **The overview's characterization of P6 is contradicted by P6.** Overview: "P6 also carries the
   program's most favorable measurement regime ... κ is large, κ/p is comfortable, and the estimator
   works as intended." P6 §6.2's own table says per-class fits are **infeasible in 6 of 10 rows**,
   including every CIFAR-100, Tiny-ImageNet and p = 2048 cell. The favorable-regime claim holds only
   for CIFAR-10 at p ≤ 512 and for the narrow-projection-head arms — a much smaller claim, and
   incidentally the one P6's `p`-sweep exists to establish rather than assume.

**Compliant and well-handled — do not change these:**

- The n = 1 / R̄ = 1 prohibition, with "per-input statistics are vMF *likelihoods*, never estimated
  κ", is stated correctly in all six sections. P6 §6.3 goes further and notes the resulting score is
  affine in cosine similarity for a fixed class — pre-empting the reviewer, which is the right move.
- The p ≥ 1024 → CUSF/torch requirement is stated in all six, with correct Bessel orders (P2 §2.7(f):
  384→191, 768→383, 1024→511, 2560→1279, 4096→2047, 12288→6143 — all correct).
- P2 §2.7(f)'s refusal to report per-layer κ̂ over `r` directions, and its choice of **trajectory
  pooling** as the primary readout because it does not degrade with model size, is the smartest
  handling of the constraint anywhere in the plan.
- P2's admission that going to 9B makes the *pooled* directional statistic **worse** (n/p ≈ 0.28,
  halved again by the block-type split) is the kind of honesty that buys credibility elsewhere.
- P4 §4.2's closed-form κ̂_null ≈ p/√n and the practitioner rule — "if your reported κ̂ is not several
  times p/√n, you have measured your sample size" — is the most portable single result in the program.
- P6 §6.2's inversion of the R̄ thresholds (0.193 at κ/p = 0.2, 0.050 at κ/p = 0.05) is correct.
- P5 §5.7 restates all three constraints correctly and adds the right distinction between the
  *encoder-predicted* κ_i (a model output, fine to report) and an *estimated* per-datapoint κ̂ (never).

---

## 6. The effective-learning-rate confound

| Project | Controlled? | Verdict |
|---|---|---|
| **P1** | Yes, comprehensively | **Fixed.** §1.1(a)–(f) is the program's reference. One gap below |
| **P2** | Yes, 8.3% of budget | **Fixed**, and it is the section's best selling point |
| **P3** | Mostly | Partial — see below |
| **P6** | **No, where it matters** | **Broken.** §1.3 |

**P1's one gap.** Section 07's ELR protocol Tier 2 specifies five arms including "(v) unnormalized +
a **scale-invariant optimizer (LARS/LAMB)** as the effective-LR-controlled reference". P1's controls
(a)–(f) include a norm-matched-WD control, a normalization-without-tangent-projection control and a
Riemannian control, but not the LARS/LAMB reference — silently dropped. It is the cheapest remaining
ELR control (one extra arm at MLP scale is ~0.2 GPU-h) and it is the one that most directly asks "can
you get the benefit from a step-size rule alone, with no sphere at all?". Add it to T1.2.

**P3's partial control.** Four LR points per arm per objective at **40% run length**, with a
full-length re-run at the selected LR — good. Two residual issues. (i) LR ranking at 40% of an RL run
is a much weaker proxy than at 40% of an SFT run, because GRPO reward curves are non-monotone and
frequently re-order late; P1 explicitly *checks* its shortened-proxy assumption (T3.5), P3 does not.
(ii) R3's mitigation — "re-running the two best LRs at full length for the two core arms (inside
T1.3/T1.4's seed budget as seeds 1–2)" — spends seeds to buy LR validation, on a design whose primary
MDE already depends on having five of them, and it covers only 2 of 4 arms. Budget the proxy check
separately (2 arms × 2 LRs × full length ≈ 25 h) or list it as a stated limitation.

**P6's failure, restated because it is the biggest ELR hole in the program.** T0.4's sweep is on
clean, 60-epoch training. The headline models are 200-epoch clean (T0.3) and 50-epoch Fast-AT
(T2.1/T2.2). Adversarial training has a materially different optimal LR from clean training, and the
normalized-vs-standard comparison the paper is *about* is made on the AT arms, which never get their
own sweep. Cost to fix: 5 LRs × 2 arms × 0.7 h ≈ **7 GPU-h**, against 16.5 h of slack. The section's
own sentence — "at this budget there is no excuse for ignoring it" — applies to itself.

---

## 7. Internal contradictions and stale numbers

**The overview (01) is the most stale document in the set and should be rewritten before anything
else is circulated.**

- Its entire "Read this first" is obsolete. Item 3 says "P3 as scoped cannot reject its own null ...
  a ~12,600 H100-hour commitment ... the fix is to buy seeds with ranks: restrict to r ∈ {1,2} and run
  **8 seeds**" — contradicting both the current P3 design (5/3 seeds, interaction demoted) and the
  program's own stated seed norms. Item 4 says "the compute plan is incoherent by 3–16× ... ~20,900
  GPU-hours against section 7's ~6,250" — fixed and no longer true. Item 2's "+7.63 headline" framing
  describes the previous P2, not the current one.
- The prose still says "Three of the **five** projects" and "The remaining projects", and the thesis
  table has four rows under a "three regimes" heading. There are six projects.
- The at-a-glance table describes P2 as "**DoLoRA replication and extension to vision** ...
  Reimplementation, mechanism diagnosis" with "the +7.63 at r=2 may not reproduce" as its biggest
  risk. P2 §2's actual framing is "does per-component strength beat a global scale", with reproduction
  explicitly relabelled *mechanism reproduction, not replication*.
- The honest-caveat paragraph says the ELR confound "threatens P1, P2, and P3 simultaneously" —
  omitting P6, where it is worst.
- The P6 measurement-regime claim contradicts P6 §6.2 (§5.5 above).

**075 (escalation) has a stale power table and a stale Tier-B table.**

- The power table lists "P3 primary **interaction** ... 1.5B, **12 seeds** ... MDE 1.6", "P3 7B tier,
  5 seeds", and "P2 language **8B** replication". All three are superseded: P3 is 4B/9B at 5/3 seeds
  with the *simple effect* as primary; P2's flagship is 9B and is labelled mechanism reproduction.
- Tier B asks are all stale: item 1 "7B, 12 seeds, ~800 H100-h" vs P3 §3.10's "Qwen3.5-9B, 2 arms × 2
  objectives × 5 seeds, **~450 H100-h**"; item 3 "8B ... ~250 H100-h" vs P2 §2.11's "~120 A100-h";
  item 4 "Olmo-3-7B ... ~220" vs P3's "~120 H100-h".
- Tier-0 entries for P3 (6.8 vs 5.2) and P6 (8.0 vs 9.9) disagree with the sections, and "All figures
  are single-A6000 hours unless noted" is false for P4, P5 and P6.

**07 (infrastructure) was not re-scoped at all and is now actively misleading.** Its header flags the
numbering mismatch, which helps, but the body still contains: an 8×A100 cluster as the hardware
assumption; D1 specified on **Gemma-3-4B**, a direct violation of the Qwen3.5-only training policy in
015; P5 (= canonical P3) at **Qwen2.5-Coder-1.5B** with "7B endpoints" and 2,500 + 2,500 H100-h; a
15-month calendar keyed to 2.1 FTE and an 8-GPU node; "P2 ... ~100–300 A100-h" for what is now a
72 + 97 3090-hour pair; and the sentence "Its compute figures ... conflict with the per-section
budgets by 3–16×", which points at a review that no longer exists. Either re-scope 07 or reduce it to
its two durable parts — the engineering-deliverables table and the eight failure modes — and delete
the sequencing, portfolio and calendar sections.

**Cross-section duplications with inconsistent estimates.**

- **The batched per-row (μ_i, κ_i) sampler + `S_outer` in-place fix** is a blocking deliverable in
  three places with three estimates: P6 §6.4 "**~1–2 days**" (and "nothing in 6.4 or 6.5 runs before
  this lands"), P5 §5.1(b) "**8 working days**, plus 2 days for the test suite", 07(a) "**2–3
  eng-weeks**". A 7× spread on an item that gates P6's Tier 1 and all of P5. Pick one owner and one
  estimate this week; it is the program's real critical path, not the GPU.
- **The differentiable log-Bessel / vMF-KL autograd layer**: P5 §5.1(a) "6 working days" vs 07(b)
  "2 + 1 eng-weeks".
- **Null tables**: P4 Tier 1 builds them (200 core-h), P6 T0.1 builds its own (0.5 h), P1 builds a
  tangent-space variant inside T0.5, and 07(e) says build once (200–400 CPU-h). P1 needs its null in
  week 1; P4 delivers in weeks 2–5. Sequence it, or P1 builds a throwaway.
- **The measurement contribution itself** is claimed as a headline by P1 §1.2, P4 §4.2/§4.4(f) and P6
  §6.2, and P4 §4.4(f) is P6 §6.2's study with `n` swept instead of `p`. P4 §4.4(f) even says it will
  "reuse Project 1's and **§6.5's** CIFAR-100 / Tiny-ImageNet ResNet-18 checkpoints" — a stale
  cross-reference to a section that did not exist when P4 was drafted.
- **The per-component-vs-global-scale question is claimed by four sections**: P2 (its entire premise,
  571 h), P3 H3 (a Gate-1 blocker), P1 §1.5 ablation 2 (vii)/(viii), and P6 §6.10 D3. P1 asserts that
  (vii)/(viii) "answers the DeLoRA-vs-DoLoRA question at **1.4 GPU-hours** on CIFAR-10 MLPs". Either
  that is true — in which case run it in week 1, before committing P2's 571 h — or it is not, in which
  case P1 should not claim it. **This is a genuine next-week action:** P1's (vii)/(viii) contrast
  (1.4 h) plus P2's T0.2 rank probe (4.6 h) cost 6 GPU-h together and decide whether the
  per-component question has any signal in either modality.

**Remaining stale model references.** 07's Gemma-3-4B and Qwen2.5-Coder-1.5B (above). P5 §5.8's
deferred text-VAE fine-tunes **GPT-2-small/medium**, a non-Qwen3.5 base, which violates 015's policy;
it is deferred, but it is not flagged as a policy exception the way P4 §4.0 and P3 §3.10 flag theirs.
P4's and P6's analysis-only diversity carve-outs are correctly argued and should stand — the Pythia
trajectory justification in particular is airtight and should be reused verbatim wherever the question
comes up.

---

## 8. What is missing — experiments a reviewer will demand that nobody proposes

**Program-level.**

1. **A dose–response test of the program's own thesis outside P3.** The thesis is a *dose* claim:
   "where the signal is scarce, the constraint pays." Only P3 §3.3 tests it as a dose (the `G` sweep
   and the dead-group-fraction knob). P1 could test the non-stationarity dose almost for free — sweep
   the number of tasks (2 / 5 / 10 / 20 classes-per-task on CIFAR-100 at MLP scale, ~2 GPU-h) and show
   the nMLP advantage growing with sequence length. Without it, P1 reports a point, and the thesis in
   the overview goes untested in its own flagship.
2. **A data-scarcity dose.** Same idea, samples-per-task rather than tasks. ~1 GPU-h at MLP scale.
3. **A pre-registration mechanism.** Only P2 §2.4 commits a success criterion to the repo with a
   commit hash and timestamp before the run. Every other section says "pre-registered" without a
   mechanism. Make P2's practice the program standard; after the matched null it is the cheapest
   credibility available.

**P6.**

4. **A common-corruption control (CIFAR-10-C) for the detector** — §1.2(b). Under 0.5 GPU-h, and the
   first thing a reviewer will ask.
5. **A held-out attack** the detector was not designed against — free.
6. **An audit of published NC1 numbers against a matched null** on released checkpoints — ~1 GPU-h,
   and it is what turns §6.2 from a rule into a result (§1.1).
7. **An LR sweep on the adversarially-trained arms** — 7 GPU-h (§6).
8. **A citation-and-delta paragraph** against the existing "neural collapse under adversarial attack /
   in adversarially trained networks" work, which already reports qualitatively what §6.6 would.

**P1.**

9. **A LARS/LAMB scale-invariant-optimizer reference arm** (07's ELR Tier-2 item (v)), dropped without
   comment. ~0.2 GPU-h.
10. **A synthetic-data throughput measurement in T3.0**, to separate the compute ceiling from the I/O
    ceiling before Gate 2→3a fires. 15 minutes (§4.2).

**P2.**

11. **AdaLoRA as a *run* arm, not a citation.** AdaLoRA's ΔW = PΛQ with an orthogonality penalty on P
    and Q is per-component scaling on approximately-normalized factors, published in 2023 and shipped
    in PEFT. §2.12 lists it under "discussed and cited but not run". A reviewer will say the novel
    delta was published three years ago with pruning attached. Adding it to the T1.2 grid at
    r ∈ {2, 32} costs ~5 GPU-h and converts the objection into a table row. The same argument applies
    more weakly to SVFT — which §2.7(c) already anticipates and includes as a diagnostic, correctly.
12. **A vision test of the "1.33× faster convergence" claim.** §2.12 measures it only at 9B. It is free
    in the VTAB grid, and wall-clock honesty is where the section is strongest.

**P3.**

13. **The ρ̂-estimation block** — 15 GPU-h (§3.2).
14. **A shortened-proxy validity check for LR selection**, analogous to P1's T3.5 — ~25 GPU-h.

**P5.**

15. **A second dataset for the dimension-wall claim.** H2 ("the uniform prior imposes a rate floor
    growing with m") is tested only on MNIST MLP VAEs, with Omniglot as a Tier-2 replication of the
    same architecture. A rate floor is a property of the prior *and* the data, and a reviewer will not
    accept a scaling law established on one binarized 28×28 dataset. CIFAR-10 at the same latent grid
    is a few GPU-hours — and cutting Tier-1 seeds from 10 to 5 (§3.6) pays for it exactly.

---

## 9. Bottom-line verdict on each project's central claim, in its current form

- **P1.** The measurement half is a real contribution but is claimed by two other sections; the
  architecture half sits at ~40% risk of being LUCIR rediscovered, which the section says out loud.
  The frontier-as-primary-object framing and the ELR suite are what make it publishable regardless of
  outcome. Fix the R̄-floor error, the CIFAR-10 n, the Gate-1 co-variation test, and the FFCV gate.
  **Still a CVPR-credible paper**, conditional on Tier 3 surviving the corrected cost model.
- **P2.** The repositioning to "does per-component strength beat a global scale" is a genuine
  improvement and makes the paper defensible, but it is a narrow within-family ablation of a published
  method, the section's own prior for Case A is **25%**, and AdaLoRA arguably occupies the ground
  already. The asset is not the delta — it is the tuned-baseline, LR-controlled, rank-resolved
  reproduction of a +7.63 claim with a built-in `r=1` null. **Lead with the reproduction; treat the
  delta as the ablation it is.** ICLR-credible on that framing; TMLR on the other.
- **P3.** Estimand and seed count are fixed; the remaining risk is entirely ρ. Gate 0 (5.2 h) and Gate
  0B (42.9 h) are cheap and decisive and should run. **Do not commit T1.3/T1.4's 187.5 h until ρ̂ is
  measured**, and re-price the 9B tier before Tier 2 is scheduled.
- **P4.** The strongest project in the program and the one with least to fix. The CRLB, the
  admissibility map, the corrected estimator, the audit, and the corrected nGPT `s_z` estimand form a
  coherent ICLR methods paper, and the head diagnostic is the only application in the plan with a
  properly computed AUC standard error. Its stated risk — "a measurement paper with no surprising
  finding" — is the right risk, and Gate C is the right gate to hang it on.
- **P5.** Solid and honest; the m ≥ 1024 numerical blocker is a real unclaimed gap and the H1/H2/H3
  separation is the science. Cut Tier-1 seeds to 5, add a second dataset, and hold the line on §5.2b —
  the LR-confound finding on Davidson's d = 40 gap would be the best outcome in the section. The
  dominant risk is correctly identified as the four developer-weeks, not the GPU.
- **P6.** The measurement half is publishable and the admissibility boundary plus the `p`-sweep is a
  real contribution — but it is a methods paper, not CVPR, and the framing ("κ̂ is a better-specified
  NC1") is not novel and under-cites SIREN and HUG. The detection half needs a Tramèr-aware framing, a
  corruption control and a held-out attack. The robustness half's key gate is mis-specified and its LR
  control misses the arms that matter. The certified-smoothing subsection is the best-governed risky
  item in the program and should be kept exactly as gated, with the input-space composition constant
  computed rather than assumed. **Fix G0(c) and the AT LR sweep before week 1, and pre-register the
  venue as a function of the Tier-0 outcome.**
