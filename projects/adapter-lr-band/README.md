# adapter-lr-band (Project 2)

Does normalizing an adapter's rank-one factors widen the range of learning rates at
which it trains successfully — and does per-component scaling add anything beyond a
single global scale?

Brief: `docs/projects/02-adapter-lr-band.md` (read with `00-shared-constraints.md`).
Pre-registration: `PREREGISTRATION.md`. Status/findings: `report.html`.

## Layout

- `src/adapters.py` — the five arms as `nn.Linear` wrappers: `lora`,
  `lora-scalar`, `lora-diag`, `delora`, `delora-pc`. All give ΔW=0 at init
  (s = softplus(λ)−log2, λ=0; plain LoRA via B=0).
- `src/effstep.py` — per-step effective-step logging (‖ΔW_{t+1}−ΔW_t‖_F/‖W₀‖_F) and
  the mutual-coherence diagnostic.
- `src/bandwidth.py` — the estimand: common grid `logspace(1e-5,1e-1,13)`, band width
  in decades at τ=1.0 abs / τ=2% rel, pre-registered divergence criterion, η_div,
  and the Falsifier-1 re-expression of bands in effective-step units.
- `src/train.py` — one run (AdamW wd0, cosine, warmup 2%, batch 32, α=2r). Real path:
  ViT-S/16 (timm) on VTAB/FGVC; `--arch tinyvit --dataset synthetic` is the local
  plumbing check only.
- `src/analyze.py` — sweep CSV → band tables + contrasts + eff-step re-expression.
- `scripts/run_grid.sh` — two-worker driver: one arm-stream per 3090.
- `tests/` — `test_identity_r1.py` asserts DeLoRA-PC ≡ DeLoRA at r=1 (forward,
  gradients, 10-step optimizer trajectory, bitwise in float64); plus arm-structure,
  estimand, and smoke tests.

## Run

```bash
uv run python -m unittest discover -s projects/adapter-lr-band/tests -v   # from repo root

# cluster, two cards, two arm-streams in parallel:
bash projects/adapter-lr-band/scripts/run_grid.sh vtab 10 1 2 4 8 16 32
uv run python projects/adapter-lr-band/src/analyze.py results/<run>/sweeps.csv
```

Do NOT run the thesis's Appendix-A singular-value-dispersion experiment
(Theorem 1 is false; see PREREGISTRATION.md).
