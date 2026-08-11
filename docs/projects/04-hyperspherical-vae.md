# Project 4 — Hyperspherical VAEs in hyperspherical architectures

**Read `00-shared-constraints.md` first.**

## The question

Davidson et al. (2018) reported that the vMF-latent VAE (S-VAE) loses to a Gaussian VAE above roughly
d=40 — the "dimension wall." **Is that wall a property of the spherical latent, or of the mismatch
between a spherical latent and an unnormalized encoder/decoder?**

The second half is the new experiment: put the vMF latent inside an architecture that is *already*
hyperspherical (nGPT-style normalization throughout — an **nViT-VAE** for images).

## Why the normalized-architecture version is the interesting one

Every S-VAE result in the literature bolts a spherical latent onto a conventional unnormalized
encoder/decoder. Davidson's own paper identifies the resulting confound without resolving it: **the
S-VAE decoder receives unit-norm inputs while the Gaussian VAE decoder receives inputs of typical
norm √d — a factor-32 difference at d=1024.** Every published S-VAE-vs-N-VAE comparison at high d
therefore compares two decoders operating at wildly different input scales, on top of whatever the
prior is doing.

A normalized architecture removes that mismatch rather than controlling for it: the encoder output,
the latent, and the decoder input all live on the sphere by construction. Nobody has run this. It
also connects this project to the normalized-network line in Projects 1 and 6.

Two enabling facts make it newly runnable: this repo has a fast batched vMF sampler and a compiled
log-Bessel path that works where SciPy fails (`ive` underflows to exactly zero above order ~511, so
**every S-VAE implementation built on SciPy is numerically invalid at m ≥ 1024**), and the Bessel
ratio `A_p(κ) = I_{p/2}(κ)/I_{p/2−1}(κ)` — which appears in both κ estimation and the S-VAE's KL
gradient — is now computable at high order.

## Fix this before Tier 1 — the gate is currently unsatisfiable

Gate G1 in v1 requires "mean learned κ falling monotonically with d **while KL still rises**." That
conjunction is satisfiable only while m ≲ 0.76κ, because **KL at fixed κ is non-monotone in m**,
peaking at m\* ≈ 0.76κ. Extrapolating Davidson's falling κ puts κ/m ≈ 1 near m ≈ 90–120, so **at
d ≥ 256 the gate is very likely mathematically unsatisfiable** — it would fail for reasons unrelated
to whether the science is right, after Tier 1 was already spent.

**Restate G1 in terms of ρ = A_m(κ), or KL/m at fixed ρ — not raw KL.**

## Asymptotics: what is true, and what they are good for

Verified to 4–5 digits numerically:
- KL(vMF(μ,κ) ‖ Uniform) = κ·A_m(κ) + log C_m(κ) + log S_{m−1}.
- At κ ≪ m: **KL ≈ κ²/(2m)** — it *vanishes* as m grows at fixed κ.
- At κ = c·m: **KL → (m/2)·log((1+√(1+4c²))/2) = Θ(m)**.
- At fixed mean resultant ρ = A_m(κ), KL is **exactly linear in m** (measured: ρ=0.3 gives
  KL/m = 0.04715 at m = 128, 1024, and 4096 identically).

**But do not use these to refute Davidson.** Inverting his reported KL column (exact, since vMF KL
depends on κ alone) gives implied κ/m of 4.6e5, 527, 82, 18.0, **4.85** at d = 2, 5, 10, 20, 40. He
never operates in the κ ≪ m regime, and his entire sweep sits on the *rising* branch of KL-vs-m. The
v1 hypothesis "KL grows roughly linearly in m at fixed κ" is a fair description of the branch the
data actually occupies. Use the asymptotics as tools for the **extension regime (d ≥ 128)**, where
the model plausibly does enter κ < m.

Two more corrections to carry:
- **"Informativeness requires κ = Θ(m)" is false.** Rate R needs only κ ≈ √(2mR) = o(m). A single
  scalar κ delivers any target rate at large m.
- The "rate-allocation bottleneck of a single scalar" diagnosis is **not novel** — it is Davidson §5.1
  verbatim ("its posterior becomes more expressive than the vMF due to the higher number of variance
  parameters"). Frame the product-of-spheres fix as testing his stated explanation at dimensions he
  could not reach, not as a new diagnosis.

## Experiments

**T1 — numerical validity (~5 GPU-h).** Show the reference S-VAE cannot produce a finite loss at
m ∈ {1024, 4096}; show this repo's path can. Reproduce Davidson's MNIST results within 2 s.d. at
d ≤ 40 as a correctness check.

**T2 — the dimension sweep, done right (~25 GPU-h).** d ∈ {5,10,20,40,64,128,256,512,1024} ×
{N-VAE, S-VAE, Power-Spherical, tied-σ N-VAE} × 5 seeds, **per-arm LR sweeps** (mandatory — v1's own
§5.2b identifies the input-scale/LR confound). Report the rate–distortion decomposition, active
units, and the gradient-variance decomposition (pathwise vs correction term).

**T3 — the nViT-VAE (the new experiment, ~30 GPU-h).** Small nViT encoder/decoder (≤20M params,
normalized throughout, learnable eigen-LRs) with a vMF latent, against four controls at matched
parameter count and per-arm tuned LR:
1. nViT encoder/decoder + **Gaussian** latent — isolates the latent from the architecture.
2. Conventional ViT encoder/decoder + **vMF** latent — the standard S-VAE setup, the mismatch case.
3. Conventional ViT + Gaussian — the plain baseline.
4. Conventional ViT + vMF + **explicit decoder input rescaling** (multiply the unit-norm latent by
   √d before the decoder) — this is the cheap control that tests whether the architecture matters
   *beyond* fixing the input-scale mismatch. **If arm 4 closes the gap, the nViT result is an
   input-scaling story and should be reported as one.**

CIFAR-10, d ∈ {64, 256, 1024}, 3 seeds. This 2×2-plus-control design is what separates "spherical
latents need spherical architectures" from "Davidson's decoder was mis-scaled."

**T4 — product-of-spheres (~10 GPU-h).** k factors = k rate knobs. Matched-rate comparison against a
diagonal Gaussian as primary, not as an ablation. If a matched-rate diagonal Gaussian closes the gap,
the win was "more rate knobs," not "spheres."

**T5 — posterior collapse, stated correctly (~15 GPU-h).** **Fixed-κ vMF is structurally immune** —
with κ fixed the rate term has zero gradient w.r.t. the encoder, so there is nothing to collapse.
**Learned-κ is not immune**: KL = 0 exactly at κ=0 with vanishing gradient there. That distinction is
the actual contribution, because fixed-κ is what Xu & Durrett (EMNLP 2018) and Guu et al. (2018) used
— precisely because learned κ was numerically unusable.

Do **not** claim "collapse is all-or-nothing in one scalar." κ is a *per-datapoint* encoder output, so
it can collapse across a subset of datapoints; and the mode that matters is **mean-map collapse**
(μ(x) → constant or confined to a low-dimensional sub-sphere) with κ untouched, where rate is paid
and no information flows. A κ histogram is blind to it and a κ-floor does not fix it — a floor buys
rate, not content. Diagnose with the **effective rank / dispersion of the aggregate {μ(x)}** or a
direct I(x;z) estimate, against a matched null.

## Datasets

MNIST (dynamically binarized) and Omniglot for the reproduction anchor only. **Add CIFAR-10 with a
small conv/ViT encoder** — a rate floor is a property of prior *and* data, and one binarized 28×28
dataset cannot carry a scaling claim. For the graph arm, keep Cora and Pubmed as reproduction
anchors, **drop Citeseer**, and add one OGB link-prediction dataset (ogbl-collab fits full-batch in
24 GB).

## Do not run

**The GPT-2/Optimus-scale text VAE.** It is 3–6× under-budgeted (a 50M decoder on WikiText-103 at
~10 epochs is ~3.5 GPU-h/run before IW-500 eval, ×90 runs ≈ 300 GPU-h, plus mandatory per-arm
annealing/free-bits tuning), it changes modality, decoder class and failure mode simultaneously — so
it does not test Davidson's wall — and m=1024 on Yelp is 30–200× oversized for a task with 5–30 nats
of usable rate, where the diagonal Gaussian will simply prune ~1000 dimensions to ~20 active units
and win for reasons unrelated to the sphere.

## Metrics

IW-500 log-likelihood with explicit rate/distortion decomposition, active units, aggregate-posterior
effective rank, gradient-variance decomposition. Honor the m = d+1 degrees-of-freedom convention
(S-VAE at latent dimension d has m = d+1 ambient coordinates) — mismatching it silently shifts every
comparison by one dimension.

## Related work to cite (currently missing)

Xu & Durrett (EMNLP 2018), Guu et al. (2018) — both used fixed-κ vMF as a text-VAE collapse fix.
Optimus (Li et al. 2020). Omitting any of these is a one-line rejection from an informed reviewer.

## Budget

~85 GPU-h total, all on a single 3090. The binding constraint is engineering time (~4 developer-weeks
for the harness), not GPU.
