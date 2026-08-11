# Coordination record — Project 5's pin of `vmf-measure` (Project 6)

**Resolved 2026-08-10.** `vmf-measure` v0.1.0 landed mid-session at commit
`2f146ef` ("frozen API/schema, log-Bessel p>=8192, refusal mode"), owner
**Andreas Plesner**, frozen schema in `projects/kappa-measure/SCHEMA.md`.
Project 5 pins **schema and signatures, not numerical values**, per the
shared constraints. This file records what Project 5 depends on; it no longer
proposes anything — the frozen API covers every need without modification.

## The pin

- All Project 5 access goes through one adapter:
  `projects/latent-geometry/src/vmf_measure_adapter.py`. It sys.path-inserts
  `projects/kappa-measure/src` (the pattern used by Project 6's own scripts),
  re-exports only what Project 5 uses, and refuses to import a different
  major version. A schema move is a one-file change.
- Smoke-verified 2026-08-10: `kappa_hat` (ok and refusal paths, frozen
  refusal string `"not resolvable at this n"`), `null_quantile` (uniform
  null, closed-form), `rel_rmse_surface`, `bessel_ratio_ap` at p = 2048.
  Sanity anchors: `rel_rmse_surface(0.85, 512, 256) = 0.0063`, reproducing
  the constraints-file canonical cell (0.63%); uniform-null κ̂ q95 ≈ 5.38 at
  (p = 512, n = 10⁴).

## Need → frozen API mapping

| Project 5 need | Frozen API |
| --- | --- |
| Per-layer κ̂_ℓ for the T1 injection scale | `kappa_hat(r_bar, p, n)` → `KappaEstimate` (joint MLE, refusal mode default) |
| Matched null for every reported κ̂ | `null_quantile("kappa_hat", p, n, q, null=...)`, enums `STATISTICS`, `NULLS` frozen |
| T2 cell admissibility annotations | `rel_rmse_surface`, `admissible`, `min_resolvable_n` |
| Dose readout E[μᵀz] = A_p(κ), p ≤ 2048 | `bessel_ratio_ap`, `bessel_ratio_ap_deriv` (valid at p ≥ 8192) |
| R̄ from layer activations | `mean_resultant_length(samples, axis=-2)` |

Notes for downstream use:

- `null_quantile` takes a **scalar** `q`; use `null_quantile_table` for grids.
- Skip `n` and you opt out of the admissibility check (`status="ok"` by
  construction). Project 5 always passes `n`.
- `nulls.py` had uncommitted working-tree changes at pin time; irrelevant to
  the pin, which covers schema/signatures only, but do not read numerical
  stability from a dirty tree.

## Table schema Project 5 will emit (T1/T2 result rows)

Aligned with SCHEMA.md §2 conventions (column order frozen once first
written; additive-only afterward). One row per (experiment, backbone, layer,
cell, seed):

```
experiment,backbone,layer,p,n,r_bar,kappa_hat,refused,null_median,null_q95,
null_q99,rel_rmse_pred,seed,commit,tool_version
```

- `refused` is true exactly when `KappaEstimate.status` is the frozen refusal
  string; `kappa_hat` is empty on those rows.
- `null_*` come from `null_quantile` at the row's own (p, n); `rel_rmse_pred`
  from `rel_rmse_surface`.
- `tool_version` mirrors SCHEMA.md (`vmf_measure.__version__`); `commit` is
  the pre-registered run config hash.

## Timing

The dependency closed before T1 execution began, so the original staging
(T1 execution without nulls, nulls by reporting time) is moot: T1 can report
with matched nulls from the start.
