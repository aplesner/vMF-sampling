# vmf-measure — frozen API and table schema (v0.1.0)

**Owner: Andreas Plesner.** Frozen end of week 2 as required by the shared
constraints. Dependent projects (1: continual learning, 5: latent-geometry
robustness) pin the **schema and API signatures below**, not the numerical
values; null quantiles and surface coefficients may be refined after the
freeze without notice, but column names, dtypes, enum values, function
signatures, and the refusal string may not change without the owner's sign-off
and a major version bump.

Package location: `projects/kappa-measure/src/vmf_measure/`.
Importable via `sys.path` insertion of that `src/` directory (see any script
in `projects/kappa-measure/scripts/` for the pattern).

## 1. Frozen Python API

### 1.1 Estimation with refusal mode — `vmf_measure.kappa`

```python
kappa_hat(r_bar: float, p: int, n: int | None = None, *,
          target_rel_rmse: float = 0.01, refuse: bool = True) -> KappaEstimate
```

`KappaEstimate` (frozen dataclass, field names and types frozen):

| field | type | meaning |
| --- | --- | --- |
| `status` | `str` | exactly `"ok"` or `"not resolvable at this n"` |
| `kappa` | `float \| None` | joint-MLE concentration; **None when refused** |
| `r_bar` | `float` | input mean resultant length |
| `p` | `int` | ambient dimension |
| `n` | `int \| None` | sample count the estimate is based on |
| `rel_rmse` | `float \| None` | predicted surface value at (r_bar, p, n) |
| `target_rel_rmse` | `float \| None` | threshold used for the refusal decision |

Refusal semantics (frozen):

- With `refuse=True` (default) and `n` given, if
  `rel_rmse_surface(r_bar, p, n) > target_rel_rmse`, return
  `status="not resolvable at this n"`, `kappa=None`. **The refusal string is
  part of the schema**; dependents may match on it.
- With `n=None`, no admissibility check is possible: `status="ok"`,
  `rel_rmse=None`. Callers that skip `n` opt out of protection.
- The estimator is the unknown-mean joint MLE (safeguarded Newton on
  `A_p(kappa) = Rbar`). `r_bar = 0` yields `kappa = 0.0`.

Also frozen: `mle_kappa(r_bar, dimension)`, `banerjee_kappa(r_bar, dimension)`
(vectorized, NumPy in/out), `mean_resultant_length(samples, axis=-2)`.

### 1.2 Admissibility calculator — `vmf_measure.admissibility`

```python
rel_rmse_surface(r_bar, dimension, n, kappa=None) -> ndarray
admissible(r_bar, dimension, n, *, target_rel_rmse=0.01) -> ndarray[bool]
min_resolvable_n(r_bar, dimension, *, target_rel_rmse=0.01, kappa=None) -> ndarray
amp(r_bar, dimension) -> ndarray
```

The surface (frozen functional form):

```
rel-RMSE ≈ hypot( amp·(1−R²)/(2nR²),  1/(κ√(n·A_p′(κ))) )
amp = 1 + 2R²/(1−R²) − 2R²/(p−R²)
```

When `kappa` is omitted it is inverted from `r_bar` via the joint MLE.
`rel_rmse_surface` returns `inf` at `r_bar <= 0` or `r_bar >= 1`.
This **replaces the rectangular rule** (`R̄ > 0.19 AND n/p ≥ 10`); the
rectangular rule must not be used for classification anywhere in the program.

### 1.3 Log-Bessel path — `vmf_measure.log_bessel`

```python
log_iv(order, x) -> ndarray                # log I_order(x), float64
bessel_ratio_ap(kappa, dimension) -> ndarray        # A_p(kappa)
bessel_ratio_ap_deriv(kappa, dimension) -> ndarray  # A_p'(kappa) = per-sample Fisher info
```

Valid at p ≥ 8192 (orders beyond 4095), where SciPy's `special.ive`
underflows to exactly zero and stock-SciPy admissibility formulas are
uncomputable. NumPy translation of the CUSF piecewise algorithm; validated
against a 50-digit mpmath reference (see `report.html`).

### 1.4 Null quantiles — `vmf_measure.nulls`

```python
null_quantile(statistic, p, n, q, *, null="uniform",
              samples=None, reps=2000, seed=None) -> NullQuantile
null_quantile_table(p_values, n_values, quantiles, *, statistics=STATISTICS,
                    null="uniform", samples=None, reps=2000, seed=None) -> list[NullQuantile]
```

Frozen enums:

- `STATISTICS = ("r_bar", "kappa_hat", "mean_cosine")`
- `NULLS = ("uniform", "mean_removed", "covariance_matched")`

`NullQuantile` fields, in `null_quantiles.csv` column order:
`statistic, null, p, n, quantile, value, method, reps, seed`
(`method ∈ {"closed_form", "monte_carlo"}`).

Null definitions (frozen):

- **uniform**: iid uniform on S^{p−1}. `r_bar` quantiles are
  `sqrt(χ²_p(q) / (n·p))` (chi-square approximation; E[R̄²] = 1/n exactly).
  `mean_cosine` = mean of `x·μ₀` against a fixed reference direction,
  Gaussian with sd `1/√(n·p)`. `kappa_hat` quantiles by monotone
  transformation of the `r_bar` quantiles. `method="closed_form"`.
- **mean_removed**: iid uniform on S^{p−1}; each observation has the
  *sample*-mean direction projected out and is renormalized to the sphere
  before the statistic is computed. Null for residual-concentration claims
  beyond one dominant direction. Monte Carlo; `r_bar` and `kappa_hat` only.
- **covariance_matched**: rows of `samples` (observed data, shape (n, p))
  supply an empirical covariance Σ; null draws are N(0, Σ) normalized to the
  sphere. Separates directional concentration from anisotropic scale.
  Monte Carlo; `r_bar` and `kappa_hat` only; `samples` required.

## 2. Frozen CSV table schemas

All CSVs are written under `projects/kappa-measure/results/`. Column **order**
is part of the schema. `tool_version` is `vmf_measure.__version__`.

### 2.1 `null_quantiles.csv`

```
statistic,null,p,n,quantile,value,method,reps,seed,tool_version
```
`statistic`,`null`,`method` use the enums above; `reps`,`seed` empty for
closed-form rows.

### 2.2 `surface_validation.csv`

One row per (grid cell, replicate batch):

```
p,kappa,kappa_over_p,n,n_over_p,reps,r_bar_mean,empirical_rel_rmse,predicted_rel_rmse,ratio_emp_over_pred,seed,sampler,tool_version
```
`empirical_rel_rmse = sqrt(mean((κ̂/κ − 1)²))` over `reps` independent
datasets; `predicted_rel_rmse` is the closed-form surface evaluated at the
population `R̄ = A_p(κ)`. Acceptance target: `ratio_emp_over_pred` within a
factor of 2 across the grid ("matches to 2 s.f.").

### 2.3 `audit_verdicts.csv`

One row per audited comparison:

```
claim_id,instrument,comparison_type,p,kappa_true_a,kappa_true_b,kappa_over_p,n_a,n_b,kappa_hat_a,kappa_hat_b,ratio_observed,ratio_true,verdict,null,reps,seed,tool_version
```

Mandatory fields per the brief (instrument, ordinal-vs-absolute, n_A vs n_B)
are covered by `instrument`, `comparison_type`, `n_a`, `n_b`.
`comparison_type ∈ {"unequal_n", "matched_n_baseline"}`.

### 2.4 Pre-registered verdict definitions (audit)

Pre-registered before any harvesting; commit hash recorded in `report.html`.

- A comparison is **audited** only if it is *absolute* (a numeric κ̂ or κ̂
  ratio enters the claim) **and** at least one arm sits at κ/p ≤ 0.2 with
  unequal n. Ordinal comparisons at matched n are excluded by design: a
  common upward bias makes them conservative.
- Verdict values (frozen enum):
  - `"not resolvable as reported at the original n"` — at least one arm's
    predicted rel-RMSE exceeds `target_rel_rmse` (0.01) at its reported n.
    This is the only adverse verdict the audit returns; **never "wrong"**.
  - `"resolvable"` — both arms admissible; the comparison stands.
  - `"out of scope"` — fails the inclusion criterion above.
- The canonical demonstration cell is κ/p = 0.05, n_A = 70 vs n_B = 700,
  identical true κ, where the spurious κ̂ ratio is 3.01.

## 3. Geometry measurement card (D4 template, frozen fields)

Every reader-facing geometry claim in the program ships with:

```
instrument | n | p | null model | observed value | null quantile (q) |
predicted rel-RMSE | admissible (y/n) | verdict
```

Rendered by `scripts/make_report.py`; see `report.html` §"Measurement card".
