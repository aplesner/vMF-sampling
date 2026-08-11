# Project 5 — Angular perturbation of classifier latents

Brief: `docs/projects/05-latent-geometry-robustness.md` (read
`docs/projects/00-shared-constraints.md` first — it wins on any disagreement).

One question: how does angular displacement propagate layer to layer under
calibrated vMF noise, and do adversarial examples displace latents *more* than
their accuracy cost predicts — or do they sit on the same dose–response curve
as ordinary corruption? Both outcomes are publishable; report whichever we get.

## Status

See `report.html` (self-contained, updated as results land).

- **Prerequisite engineering — done (2026-08-10).** Per-row `(mu_i, kappa_i)`
  batched sampling landed in `src/vmf_torch_batched.py` (`sample_vmf_rows`)
  with statistical validation in `tests/test_batched.py`. The binding
  constraint from the brief is removed; the real cost was under one day,
  against a briefed range of 2 days to 3 weeks.
- **Dependency resolved (2026-08-10):** Project 6's `vmf-measure` v0.1.0
  landed at commit `2f146ef` with a frozen API/schema (owner: Andreas
  Plesner). Project 5 pins it through one adapter
  (`src/vmf_measure_adapter.py`); the pin is recorded in
  `vmf_measure_interface.md`.

## Layout

- `src/` (project-local experiment code, as it lands)
- `configs/` (run configs; pre-registration commit hashes go here)
- `results/` (small result tables only — checkpoints and raw logs live on
  `net_scratch`, never here)
- `report.html` — primary channel for findings
- `vmf_measure_interface.md` — frozen-schema coordination note for Project 6

## Guardrails carried from the brief and constraints file

- Do **not** reintroduce the two retired v1 claims: per-class κ̂ at CIFAR-100
  scale is well-conditioned (measured rel. RMSE 0.63%), and the KappaFace
  audit target is a category error (Spearman 0.995; κ̂ is z-scored across
  classes, so common bias cancels by construction).
- Every κ̂ ships with a matched null from `vmf-measure` — do not write local
  null calibration.
- Detection (T3) is Tramèr-compliant or not at all: shared encoders, LID and
  feature squeezing as baselines, CIFAR-10-C control at matched clean-accuracy
  degradation, held-out attacks (AutoAttack + Square, split declared before
  running), and post-adaptive AUROC as the headline.
- T4 has a pre-registered kill condition: gap below MDE (~2.5–4.7 points at
  3 seeds) at tuned LRs, or any masking diagnostic failing in ≥2/3 seeds →
  the intervention arm ships as a falsification paragraph.
