# Research program: representations on the hypersphere

Six projects built on this repo's fast vMF sampler and its numerically valid high-order log-Bessel
path. Each is scoped as a **runnable experiment with one question**, not as a paper pitch. Venues get
decided from results.

## Read this first

- **`projects/00-shared-constraints.md` is authoritative.** Compute envelope, measurement rules,
  statistical practice, and — most importantly — a table of **claims retired after numerical
  verification**. Several confident-sounding results in the first version of this program were
  checked and found false. Read that table before proposing anything.
- **One question per project. One question per paper.** Do not bundle a measurement contribution, a
  method contribution, and an application into one write-up.
- **Do not design for a venue.** Venue-first thinking produced the discarded v1.
- Project 6 is a **dependency** for Projects 1 and 5. It runs first.

## The projects

| # | Project | Question | Budget |
| --- | --- | --- | --- |
| [1](projects/01-continual-learning.md) | Continual learning | Does architecture-wide normalization improve the stability–plasticity frontier in rehearsal-free CIL, and is bounded angular motion the reason? | ~400 A6000-h |
| [2](projects/02-adapter-lr-band.md) | Adapter LR band | Does normalizing an adapter's rank-one factors widen the usable learning-rate range, and does per-component scaling add anything? | ~110 |
| [3](projects/03-rlvr-code.md) | RLVR on code | Does the adapter advantage grow as the reward carries fewer bits? | ~200 |
| [4](projects/04-hyperspherical-vae.md) | Hyperspherical VAE | Is the S-VAE dimension wall a property of the spherical latent, or of its mismatch with an unnormalized architecture? | ~85 |
| [5](projects/05-latent-geometry-robustness.md) | Angular perturbation | Do adversarial examples displace latents more than their accuracy cost predicts, or sit on the corruption curve? | ~40 |
| [6](projects/06-kappa-measurement.md) | κ measurement | When is a geometry measurement resolvable at all, and what does a practitioner do about it? | ~40 + 150 CPU-core-h |

Roughly 875 GPU-h across two card types. Every project's first gate opens for under 15 GPU-h.

## What changed from v1

v1 (archived at `archive/research_program-v1-superseded.md`, 77k words) was reviewed by an
independent pass and then adversarially verified. The verification pass ran numerics rather than
arguments, and it overturned six substantive claims — including two that had been reported as
headline findings. The retired-claims table in `00-shared-constraints.md` records each with its
correction.

Structural changes:

- **Projects 2 and 3 stay separate.** A merge was proposed and rejected: the merged primary estimand
  confounds normalization (published as DeLoRA) with per-component scaling (the untested delta), and
  the merged escape hatch lands on a design already labelled underpowered.
- **Project 2's headline moved** from an accuracy delta to the learning-rate band — an estimand that
  is immune to the undertuned-baseline attack by construction and survives an accuracy null, which
  is the modal outcome.
- **Project 4 gained the experiment that motivates it**: a vMF latent inside a *normalized*
  architecture (nViT-VAE), which removes the encoder/decoder scale mismatch Davidson identified and
  never resolved, instead of controlling for it.
- **Project 5 was cut from four claims to one**, and its intended headline negative result was found
  to be arithmetically false.
- **Project 6 was cut from a literature census to an instrument paper with a narrow audit.** The
  yield model gives 0–2 consequential findings, not a flagship.
- **Two bugs were caught before any spend:** a train/test leak in Project 3 (trains on
  LiveCodeBench-v6, evaluates on "LCB-v6 easy"), and Project 4's Gate G1 being mathematically
  unsatisfiable at d ≥ 256.

## Provenance notes

`papers/nLoRA.pdf` ("DoLoRA") is a **master's thesis supervised by the project owner** — never
published, never submitted, code not retained, **SFT only**. Replication is a genuine prerequisite,
not a rhetorical exercise. Do not infer anything from its being unpublished.

## Unresolved

Several projects version-pin the same public artifact (`vmf-measure`). Three concurrent double-blind
submissions pinning one public repo de-anonymizes the group and reads as salami to a reviewer who
checks; anonymized release breaks the pin-and-cite workflow. This needs an answer before any
double-blind submission, and it does not have one yet.
