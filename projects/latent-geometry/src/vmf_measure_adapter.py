"""Single adapter isolating every Project 5 call into Project 6's vmf-measure.

Pinned: vmf-measure v0.1.0, frozen schema in
``projects/kappa-measure/SCHEMA.md`` (owner: Andreas Plesner), base commit
``2f146ef``. Dependents pin the schema and signatures, not numerical values.
If the package location moves, this file is the only Project 5 change.

The package is imported via ``sys.path`` insertion of its ``src/`` directory,
the same pattern used by ``projects/kappa-measure/scripts/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VMF_MEASURE_SRC = (
    Path(__file__).resolve().parents[2] / "kappa-measure" / "src"
)
if str(_VMF_MEASURE_SRC) not in sys.path:
    sys.path.insert(0, str(_VMF_MEASURE_SRC))

import vmf_measure  # noqa: E402
from vmf_measure import (  # noqa: E402
    NULLS,
    STATISTICS,
    KappaEstimate,
    NullQuantile,
    admissible,
    bessel_ratio_ap,
    bessel_ratio_ap_deriv,
    kappa_hat,
    mean_resultant_length,
    null_quantile,
    null_quantile_table,
    rel_rmse_surface,
)

_EXPECTED_VERSION_MAJOR = "0"

if vmf_measure.__version__.split(".")[0] != _EXPECTED_VERSION_MAJOR:
    raise ImportError(
        f"Project 5 pins vmf-measure schema v{_EXPECTED_VERSION_MAJOR}.x; "
        f"found {vmf_measure.__version__}. Re-check SCHEMA.md before upgrading."
    )

__all__ = [
    "NULLS",
    "STATISTICS",
    "KappaEstimate",
    "NullQuantile",
    "admissible",
    "bessel_ratio_ap",
    "bessel_ratio_ap_deriv",
    "kappa_hat",
    "mean_resultant_length",
    "null_quantile",
    "null_quantile_table",
    "rel_rmse_surface",
]
