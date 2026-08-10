"""vmf-measure: feasibility, nulls, and a refusing estimator for vMF geometry.

Project 6 deliverable D2.  The public API and the CSV table schemas are
frozen in ``projects/kappa-measure/SCHEMA.md``; dependents pin the schema,
not the numerical values.

- :func:`kappa_hat` — concentration estimate with refusal mode as default
  (returns "not resolvable at this n" below the admissibility boundary).
- :func:`rel_rmse_surface` / :func:`admissible` / :func:`min_resolvable_n` —
  the closed-form feasibility surface that replaces the rectangular rule.
- :func:`null_quantile` / :func:`null_quantile_table` — uniform,
  mean-removed, and covariance-matched null quantiles.
- :func:`log_iv` / :func:`bessel_ratio_ap` — log-Bessel path valid at
  p >= 8192, where SciPy's ``ive`` underflows.
"""

from .admissibility import admissible, amp, min_resolvable_n, rel_rmse_surface
from .kappa import (
    DEFAULT_TARGET_REL_RMSE,
    OK_STATUS,
    REFUSAL_STATUS,
    KappaEstimate,
    banerjee_kappa,
    kappa_hat,
    mean_resultant_length,
    mle_kappa,
)
from .log_bessel import bessel_ratio_ap, bessel_ratio_ap_deriv, log_iv
from .nulls import NULLS, STATISTICS, NullQuantile, null_quantile, null_quantile_table

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TARGET_REL_RMSE",
    "NULLS",
    "OK_STATUS",
    "REFUSAL_STATUS",
    "STATISTICS",
    "KappaEstimate",
    "NullQuantile",
    "__version__",
    "admissible",
    "amp",
    "banerjee_kappa",
    "bessel_ratio_ap",
    "bessel_ratio_ap_deriv",
    "kappa_hat",
    "log_iv",
    "mean_resultant_length",
    "min_resolvable_n",
    "mle_kappa",
    "null_quantile",
    "null_quantile_table",
    "rel_rmse_surface",
]
