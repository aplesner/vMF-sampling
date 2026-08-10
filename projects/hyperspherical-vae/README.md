# Project 4 — hyperspherical VAEs in hyperspherical architectures

Brief: `docs/projects/04-hyperspherical-vae.md` (read
`docs/projects/00-shared-constraints.md` first). Report: `report.html`
(regenerate with `uv run python scripts/make_report.py`).

## Layout

- `PREREGISTRATION.md` — restated Gate G1 and pre-registered success criteria.
- `src/vmf_kl.py` — float64 vMF KL / `A_m(κ)` / inversions, valid at m ≥ 1024
  (CUSF/torch log-Bessel path; large-κ asymptotic branch). This is the
  numerics core every tier builds on.
- `scripts/verify_g1.py` — V1–V5 verification of the restated gate; writes
  `results/g1_*.csv` and `results/g1_verification_log.txt`.
- `scripts/make_report.py` — regenerates `report.html` from `results/`.
- `tests/` — `uv run python -m unittest discover -s tests -v` from this
  directory.

## Status

Gate G1 fixed and verified (2026-07-27). T1–T5 not started. GPU runs target
two RTX 3090s as independent workers (`--gres=gpu:geforce_rtx_3090:1` on
tikgpu[06,07,09]); one (arm × d × seed) run per card.
