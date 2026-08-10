"""Closed-form feasibility surface for vMF concentration measurement.

Implements deliverable D1 of the Project 6 brief:

    rel-RMSE(kappa_hat) ~= hypot( amp*(1-R^2)/(2*n*R^2),  1/(kappa*sqrt(n*A_p'(kappa))) )
    amp = 1 + 2*R^2/(1-R^2) - 2*R^2/(p-R^2)

This replaces the rectangular rule (Rbar > 0.19 AND n/p >= 10), which is
grossly conservative in the high-concentration corner and produces false
positives when used for audit classification.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .log_bessel import bessel_ratio_ap, bessel_ratio_ap_deriv


def amp(r_bar: ArrayLike, dimension: ArrayLike) -> NDArray[np.float64]:
    """Amplification factor of the admissibility surface."""

    r, p = np.broadcast_arrays(
        np.asarray(r_bar, dtype=np.float64), np.asarray(dimension, dtype=np.float64)
    )
    r2 = r * r
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 + 2.0 * r2 / (1.0 - r2) - 2.0 * r2 / (p - r2)


def rel_rmse_surface(
    r_bar: ArrayLike,
    dimension: ArrayLike,
    n: ArrayLike,
    kappa: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Predicted relative RMSE of the joint-MLE kappa estimator.

    ``r_bar`` is the (sample or population) mean resultant length, ``n`` the
    sample count it was computed from, and ``kappa`` the concentration.  When
    ``kappa`` is omitted it is inverted from ``r_bar`` with the safeguarded
    Newton solver (unknown-mean joint MLE), which is the intended usage for
    the admissibility calculator.

    Returns inf where the surface is undefined (r_bar <= 0 or r_bar >= 1).
    """

    from .kappa import mle_kappa  # local import: kappa imports nothing here

    r, p, n_arr = np.broadcast_arrays(
        np.asarray(r_bar, dtype=np.float64),
        np.asarray(dimension, dtype=np.float64),
        np.asarray(n, dtype=np.float64),
    )
    if np.any(n_arr <= 0.0):
        raise ValueError("n must be positive")
    if kappa is None:
        kappa_arr = mle_kappa(r, p)
    else:
        kappa_arr = np.broadcast_to(np.asarray(kappa, dtype=np.float64), r.shape)

    r2 = r * r
    with np.errstate(divide="ignore", invalid="ignore"):
        stochastic = amp(r, p) * (1.0 - r2) / (2.0 * n_arr * r2)
        fisher = 1.0 / (kappa_arr * np.sqrt(n_arr * bessel_ratio_ap_deriv(kappa_arr, p)))
        surface = np.hypot(stochastic, fisher)
    undefined = (r <= 0.0) | (r >= 1.0) | (kappa_arr <= 0.0)
    return np.where(undefined, np.inf, surface)


def admissible(
    r_bar: ArrayLike,
    dimension: ArrayLike,
    n: ArrayLike,
    *,
    target_rel_rmse: float = 0.01,
) -> NDArray[np.bool_]:
    """True where the predicted relative RMSE meets ``target_rel_rmse``."""

    return rel_rmse_surface(r_bar, dimension, n) <= target_rel_rmse


def min_resolvable_n(
    r_bar: ArrayLike,
    dimension: ArrayLike,
    *,
    target_rel_rmse: float = 0.01,
    kappa: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Smallest n at which the surface meets ``target_rel_rmse``.

    The surface is strictly decreasing in n, so a bisection on log n is
    exact to the tolerance of the bracket.  Returns inf where no finite n
    suffices (r_bar <= 0 or r_bar >= 1).
    """

    r, p = np.broadcast_arrays(
        np.asarray(r_bar, dtype=np.float64), np.asarray(dimension, dtype=np.float64)
    )
    if kappa is None:
        kappa_arg = None
    else:
        kappa_arg = np.broadcast_to(np.asarray(kappa, dtype=np.float64), r.shape)

    result = np.full(r.shape, np.inf)
    valid = (r > 0.0) & (r < 1.0)
    if not np.any(valid):
        return result

    lo = np.zeros(r.shape)  # log10 n lower bound (n = 1)
    hi = np.full(r.shape, 12.0)  # log10 n upper bound (n = 1e12)
    # Extend the upper bracket until every valid cell is below target.
    for _ in range(64):
        surface_hi = rel_rmse_surface(
            np.where(valid, r, 0.5), p, 10.0**hi, kappa=kappa_arg
        )
        grow = valid & (surface_hi > target_rel_rmse)
        if not np.any(grow):
            break
        hi = np.where(grow, hi + 4.0, hi)
    else:
        return result

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        surface_mid = rel_rmse_surface(
            np.where(valid, r, 0.5), p, 10.0**mid, kappa=kappa_arg
        )
        below = surface_mid <= target_rel_rmse
        hi = np.where(valid & below, mid, hi)
        lo = np.where(valid & ~below, mid, lo)

    # Return the upper bracket endpoint: it is certified to meet the target.
    result = np.where(valid, 10.0**hi, result)
    return result
